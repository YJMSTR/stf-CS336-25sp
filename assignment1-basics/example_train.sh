#!/bin/bash
# Example training script for the Transformer Language Model

# This script demonstrates how to use the training script with various configurations.
# Adjust the parameters according to your needs and hardware capabilities.

# Small model configuration for testing
echo "Training a small Transformer model..."

python train.py \
    --train_data tests/fixtures/tinystories_sample_5M.npy \
    --val_data tests/fixtures/tinystories_sample_5M.npy \
    --vocab_size 10000 \
    --context_length 256 \
    --d_model 256 \
    --num_layers 4 \
    --num_heads 8 \
    --d_ff 1024 \
    --rope_theta 10000.0 \
    --batch_size 32 \
    --max_iters 10000 \
    --learning_rate 3e-4 \
    --min_learning_rate 3e-5 \
    --warmup_iters 500 \
    --weight_decay 0.1 \
    --grad_clip 1.0 \
    --checkpoint_dir checkpoints/small_model \
    --checkpoint_interval 1000 \
    --log_interval 50 \
    --eval_interval 500 \
    --eval_iters 50 \
    --device cuda \
    --dtype float16 \
    --seed 42

# Medium model configuration (similar to GPT-2 small)
# Uncomment to use:
# python train.py \
#     --train_data path/to/train_data.npy \
#     --val_data path/to/val_data.npy \
#     --vocab_size 50257 \
#     --context_length 1024 \
#     --d_model 768 \
#     --num_layers 12 \
#     --num_heads 12 \
#     --d_ff 3072 \
#     --rope_theta 10000.0 \
#     --batch_size 16 \
#     --max_iters 100000 \
#     --learning_rate 6e-4 \
#     --min_learning_rate 6e-5 \
#     --warmup_iters 2000 \
#     --weight_decay 0.1 \
#     --grad_clip 1.0 \
#     --checkpoint_dir checkpoints/medium_model \
#     --checkpoint_interval 5000 \
#     --log_interval 100 \
#     --eval_interval 1000 \
#     --eval_iters 100 \
#     --device cuda \
#     --dtype float16 \
#     --seed 42

# Resume training from checkpoint
# Uncomment to use:
# python train.py \
#     --train_data path/to/train_data.npy \
#     --val_data path/to/val_data.npy \
#     --resume_from checkpoints/medium_model/checkpoint_latest.pt \
#     ... (other parameters same as above)

# Training with Weights & Biases logging
# Uncomment to use:
# python train.py \
#     --train_data path/to/train_data.npy \
#     --val_data path/to/val_data.npy \
#     --wandb_project transformer_experiments \
#     --wandb_run_name small_model_run1 \
#     ... (other parameters)
