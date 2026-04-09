#!/bin/bash
# Qwen3.5-35B-A3B (MoE: 35B total, 3B active) Icelandic continued pretraining with LoQT
# Hardware: 2x H100 80GB

# Use GPU 1 if GPU 0 is busy (interactive node sharing)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

export HF_HOME=/dtu/p1/vestsn/isft/hf_cache
export HF_DATASETS_CACHE=/dtu/p1/vestsn/isft/hf_cache/datasets
export TMPDIR=/dtu/p1/vestsn/isft/tmp
export TEMP=/dtu/p1/vestsn/isft/tmp
export TMP=/dtu/p1/vestsn/isft/tmp
export XDG_CACHE_HOME=/dtu/p1/vestsn/isft/cache
export WANDB_DIR=/dtu/p1/vestsn/isft/LoQT
export WANDB_CACHE_DIR=/dtu/p1/vestsn/isft/cache/wandb
export HF_TOKEN=$(python3 -c "from huggingface_hub import get_token; print(get_token())")
export QUANTIZED_MODEL_PATH=/dtu/p1/vestsn/isft/quantized_model.pt
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$WANDB_CACHE_DIR"

PYTHONUNBUFFERED=1 python3 -m torch.distributed.run --standalone --nproc_per_node 1 --master_port 29500 torchrun_main.py \
    --model_name Qwen/Qwen3.5-35B-A3B \
    --dataset_name /dtu/p1/vestsn/isft/icelandic_data \
    --use_hf_model True \
    --lr 0.0005 \
    --rank 64 \
    --lora_alpha 0.5 \
    --update_proj_gap 100 \
    --batch_size 4 \
    --total_batch_size 64 \
    --max_length 2048 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --eval_every 1000 \
    --save_every 20 \
    --dtype bfloat16 \
    --optimizer adam8bit \
    --use_loqt True \
    --quantize_w 4bit \
    --quantize_projection_matrix '4bit' \
    --use_double_quant True \
    --bnb_4bit_quant_type nf4 \
    --compensate_quant_error_iterations 5 \
    --proj_gap_progression exponential \
    --increment_size 1.2 \
    --target_modules attn shared_expert \
    --quantize_frozen_experts True \
    --use_offloading True \
    --activation_checkpointing \
    --seed 42 \
    --save_dir checkpoints/qwen35_35b_icelandic \
    --name qwen35_35b_icelandic_loqt \
    --wandb_project icelandic-loqt \
    --use_chat_template True \
    --is_icelandic_dataset True \
    --workers 0
