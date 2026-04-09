"""Utilities for making fused MoE expert weights compatible with LoQT.

MoE models like Qwen3.5 store expert weights as 3D nn.Parameter tensors
for efficient batched computation. LoQT can only wrap nn.Linear modules.
This module provides utilities to "unfuse" expert weights into individual
nn.Linear layers that LoQT can wrap, while keeping the same forward semantics.

Also provides quantization of frozen expert weights to NF4 for memory savings
when experts are not being trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes.functional as bnb_F


class UnfusedMoeExperts(nn.Module):
    """MoE experts stored as individual nn.Linear layers instead of fused 3D Parameters.

    Drop-in replacement for Qwen3_5MoeExperts with the same forward signature.
    Each expert's gate_up_proj and down_proj become separate nn.Linear modules,
    making them wrappable by LoQT.
    """

    def __init__(self, num_experts, hidden_dim, intermediate_dim, act_fn):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.act_fn = act_fn

        # Individual linear layers per expert — LoQT can wrap each one
        # Use device='meta' to skip random weight initialization (we copy weights after)
        self.gate_up = nn.ModuleList([
            nn.Linear(hidden_dim, 2 * intermediate_dim, bias=False, device='meta')
            for _ in range(num_experts)
        ])
        self.down = nn.ModuleList([
            nn.Linear(intermediate_dim, hidden_dim, bias=False, device='meta')
            for _ in range(num_experts)
        ])

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            gate_up_out = self.gate_up[expert_idx](current_state)
            gate, up = gate_up_out.chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = self.down[expert_idx](current_hidden_states)

            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


def unfuse_moe_experts(model):
    """Replace fused 3D-Parameter MoE expert modules with unfused nn.Linear layers.

    Finds modules with 3D gate_up_proj/down_proj Parameters (like Qwen3_5MoeExperts)
    and replaces them with UnfusedMoeExperts containing individual nn.Linear layers.
    This makes expert weights wrappable by LoQT without changing the model's behavior.

    Args:
        model: The HuggingFace model to modify in-place.

    Returns:
        The modified model.
    """
    from transformers.activations import ACT2FN

    replacements = []

    # Collect modules to replace (can't modify during iteration)
    for name, module in model.named_modules():
        if not (hasattr(module, 'gate_up_proj') and hasattr(module, 'down_proj')):
            continue
        gate_up = getattr(module, 'gate_up_proj')
        down = getattr(module, 'down_proj')
        if not (isinstance(gate_up, nn.Parameter) and gate_up.dim() == 3):
            continue
        replacements.append((name, module, gate_up, down))

    for name, module, gate_up, down in replacements:
        num_experts, intermediate_dim_2, hidden_dim = gate_up.shape
        intermediate_dim = intermediate_dim_2 // 2
        # down_proj shape: (num_experts, hidden_dim, intermediate_dim)

        act_fn = module.act_fn if hasattr(module, 'act_fn') else ACT2FN['silu']

        unfused = UnfusedMoeExperts(num_experts, hidden_dim, intermediate_dim, act_fn)

        # Copy weights from 3D tensors to individual nn.Linear layers
        # Materialize from meta device and copy in one step
        with torch.no_grad():
            for i in range(num_experts):
                unfused.gate_up[i].weight = nn.Parameter(gate_up.data[i].clone())
                unfused.down[i].weight = nn.Parameter(down.data[i].clone())

        # Replace in parent module
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], unfused)

    if replacements:
        # Free memory from old 3D tensors
        torch.cuda.empty_cache()
        print(f"Unfused {len(replacements)} MoE expert modules "
              f"({replacements[0][2].shape[0]} experts each) into individual nn.Linear layers")

    return model


class QuantizedMoeExperts(nn.Module):
    """MoE experts with NF4-quantized frozen weights.

    Drop-in replacement for Qwen3_5MoeExperts. Stores each expert's 2D weight
    slice as NF4-quantized data, dequantizes on-the-fly during forward.
    Reduces expert memory from bf16 (2 bytes/param) to NF4 (~0.5 bytes/param).
    """

    def __init__(self, num_experts, hidden_dim, intermediate_dim, act_fn,
                 gate_up_quant_data, gate_up_quant_states,
                 down_quant_data, down_quant_states):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.act_fn = act_fn

        # Store quantized data as buffers (not parameters — frozen)
        for i in range(num_experts):
            self.register_buffer(f'gate_up_qw_{i}', gate_up_quant_data[i])
            self.register_buffer(f'down_qw_{i}', down_quant_data[i])

        # Store quant states in a plain list (not nn.Module managed)
        self.gate_up_qs = gate_up_quant_states
        self.down_qs = down_quant_states

    def _move_qs_to(self, device):
        """Move all quant_state tensors to device."""
        for qs_list in [self.gate_up_qs, self.down_qs]:
            for qs in qs_list:
                qs.absmax = qs.absmax.to(device)
                if qs.code is not None:
                    qs.code = qs.code.to(device)
                if qs.state2 is not None:
                    qs.state2.absmax = qs.state2.absmax.to(device)
                    if qs.state2.code is not None:
                        qs.state2.code = qs.state2.code.to(device)

    def _apply(self, fn, recurse=True):
        """Override _apply to also move quant_state tensors when model.to() is called."""
        result = super()._apply(fn, recurse)
        # Detect target device from the first buffer
        buf = getattr(self, 'gate_up_qw_0', None)
        if buf is not None:
            self._move_qs_to(buf.device)
        return result

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        device = args[0] if args and isinstance(args[0], (torch.device, str, int)) else kwargs.get('device', None)
        if device is not None:
            self._move_qs_to(device)
        return result

    def _dequant(self, idx, which):
        if which == 'gate_up':
            qw = getattr(self, f'gate_up_qw_{idx}')
            qs = self.gate_up_qs[idx]
        else:
            qw = getattr(self, f'down_qw_{idx}')
            qs = self.down_qs[idx]
        return bnb_F.dequantize_4bit(qw, qs, quant_type='nf4')

    def forward(self, hidden_states, top_k_index, top_k_weights):
        device = hidden_states.device
        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            gate_up_w = self._dequant(expert_idx.item(), 'gate_up').to(current_state.dtype)
            gate, up = F.linear(current_state, gate_up_w).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up

            down_w = self._dequant(expert_idx.item(), 'down').to(current_state.dtype)
            current_hidden_states = F.linear(current_hidden_states, down_w)

            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx.to(device), current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


def quantize_frozen_experts(model):
    """Quantize MoE expert weights to NF4 in-place (frozen, not trained).

    Replaces Qwen3_5MoeExperts modules with QuantizedMoeExperts that store
    weights in NF4 format. Reduces expert memory by ~4x.
    """
    replacements = []

    for name, module in model.named_modules():
        if not (hasattr(module, 'gate_up_proj') and hasattr(module, 'down_proj')):
            continue
        gate_up = getattr(module, 'gate_up_proj')
        down = getattr(module, 'down_proj')
        if not (isinstance(gate_up, nn.Parameter) and gate_up.dim() == 3):
            continue
        replacements.append((name, module, gate_up, down))

    for name, module, gate_up, down in replacements:
        num_experts, intermediate_dim_2, hidden_dim = gate_up.shape
        intermediate_dim = down.shape[2]  # (num_experts, hidden_dim, intermediate_dim)

        act_fn = module.act_fn if hasattr(module, 'act_fn') else nn.SiLU()

        # Quantize each expert's 2D weight to NF4
        gate_up_qdata = []
        gate_up_qstates = []
        down_qdata = []
        down_qstates = []

        # Use GPU for quantization if available (CPU bnb backend is ~100x slower)
        quant_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        with torch.no_grad():
            for i in range(num_experts):
                qw, qs = bnb_F.quantize_4bit(gate_up.data[i].to(device=quant_device, dtype=torch.float32), quant_type='nf4')
                qs.shape = torch.Size(qs.shape)  # Fix bnb 0.46 shape bug
                gate_up_qdata.append(qw.cpu())
                gate_up_qstates.append(qs)

                qw, qs = bnb_F.quantize_4bit(down.data[i].to(device=quant_device, dtype=torch.float32), quant_type='nf4')
                qs.shape = torch.Size(qs.shape)
                down_qdata.append(qw.cpu())
                down_qstates.append(qs)

                # Move quant_state tensors back to CPU to free GPU
                qs.absmax = qs.absmax.cpu()
                if qs.code is not None:
                    qs.code = qs.code.cpu()
                if qs.state2 is not None:
                    qs.state2.absmax = qs.state2.absmax.cpu()
                gate_up_qstates[-1].absmax = gate_up_qstates[-1].absmax.cpu()

            if quant_device != 'cpu':
                torch.cuda.empty_cache()

        quantized = QuantizedMoeExperts(
            num_experts, hidden_dim, intermediate_dim, act_fn,
            gate_up_qdata, gate_up_qstates,
            down_qdata, down_qstates,
        )

        # Replace in parent module
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], quantized)

    if replacements:
        import gc; gc.collect()
        print(f"Quantized {len(replacements)} MoE expert modules to NF4 "
              f"({replacements[0][2].shape[0]} experts each, ~4x memory reduction)")

    return model
