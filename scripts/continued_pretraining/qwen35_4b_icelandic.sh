#!/bin/bash
# Qwen3.5-4B (dense, DeltaNet hybrid) Icelandic continued pretraining with LoQT
# Hardware: single GPU (RTX 3090 24GB or similar)

echo "Starting script"

# Cache dirs — adjust for your local machine
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export TMPDIR=${TMPDIR:-/tmp}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHONUNBUFFERED=1 python3 -m torch.distributed.run --standalone --nproc_per_node 1 --master_port 29500 torchrun_main.py \
    --model_name Qwen/Qwen3.5-4B \
    --dataset_name /dtu/p1/vestsn/isft/icelandic_data \
    --use_hf_model True \
    --lr 0.0005 \
    --rank 128 \
    --lora_alpha 0.5 \
    --update_proj_gap 100 \
    --batch_size 2 \
    --total_batch_size 32 \
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
    --target_modules attn mlp \
    --use_offloading True \
    --activation_checkpointing \
    --seed 42 \
    --save_dir checkpoints/qwen35_4b_icelandic \
    --name qwen35_4b_icelandic_loqt \
    --wandb_project icelandic-loqt \
    --use_chat_template True \
    --is_icelandic_dataset True \
    --workers 0
