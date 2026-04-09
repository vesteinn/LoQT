"""Quick smoke test: load Qwen3.5-35B-A3B, apply LoQT 4-bit, run one forward pass."""
import os
os.environ["HF_HOME"] = "/dtu/p1/vestsn/isft/hf_cache"

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from loqt.LoQT import LoQTModel

cache_dir = "/dtu/p1/vestsn/isft/hf_cache"
model_name = "Qwen/Qwen3.5-35B-A3B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model (bf16, CPU first)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
print(f"Model type: {type(model).__name__}")
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

# Count nn.Linear layers that LoQT would target
target = ["attn", "mlp"]
linear_count = 0
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and any(t in name for t in target):
        linear_count += 1
print(f"nn.Linear layers matching ['attn', 'mlp']: {linear_count}")

device = torch.device("cuda:0")

print("\nWrapping with LoQT (4-bit quantization, rank=256)...")
model = LoQTModel(
    model,
    r=256,
    lora_alpha=0.5,
    target_modules=["attn", "mlp"],
    quantize_w="4bit",
    use_double_quant=True,
    device=device,
    proj_type="std",
    compute_dtype=torch.bfloat16,
    quantize_projection_matrix="4bit",
    compensate_quant_error_iterations=0,
    is_single_gpu=True,
    only_train_lora=False,
    model_config=model.config.to_dict() if hasattr(model.config, 'to_dict') else {},
    use_eigenh_for_projection=False,
    init_lora_AB_as_random_and_zeros=False,
    train_projection_matrix=False,
    update_steps=[100, 200],
    grad_accumulation_steps=1,
)

print("Enabling gradient checkpointing...")
model.wrapped_model.gradient_checkpointing_enable()

print("Moving to GPU...")
model = model.to(device=device, dtype=torch.bfloat16)

# Memory report
allocated = torch.cuda.memory_allocated(0) / 1e9
reserved = torch.cuda.memory_reserved(0) / 1e9
print(f"GPU memory: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

print("\nRunning forward pass...")
text = "Ísland er eyja í norðanverðum Atlantshafi."
inputs = tokenizer(text, return_tensors="pt", max_length=32, truncation=True, padding="max_length").to(device)
labels = inputs["input_ids"].clone()
labels[labels == tokenizer.pad_token_id] = -100

with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    output = model(**inputs, labels=labels)

print(f"Loss: {output.loss.item():.4f}")
print(f"GPU memory after forward: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")

print("\nRunning backward pass...")
output.loss.backward()
print(f"GPU memory after backward: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")

print("\nSMOKE TEST PASSED")
