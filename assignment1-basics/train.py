#!/usr/bin/env python3
"""
Training script for Transformer Language Model.
Supports configurable hyperparameters, memory-efficient data loading,
checkpointing, and logging to console/Weights & Biases.
"""

import os
import argparse
import time
import math
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from pathlib import Path
from datetime import datetime
import json
import sys

# Add the current directory to the path so we can import cs336_basics
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cs336_basics.transformer import (
    TransformerLM, 
    cross_entropy, 
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    save_checkpoint,
    load_checkpoint
)

from experiment_tracker import ExperimentTracker

# Optional: Weights & Biases support
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Install with 'pip install wandb' for experiment tracking.")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train a Transformer Language Model')
    
    # Model hyperparameters
    parser.add_argument('--vocab_size', type=int, default=10000,
                        help='Size of the vocabulary')
    parser.add_argument('--context_length', type=int, default=128,
                        help='Maximum context length')
    parser.add_argument('--d_model', type=int, default=768,
                        help='Hidden dimension of the model')
    parser.add_argument('--num_layers', type=int, default=12,
                        help='Number of transformer blocks')
    parser.add_argument('--num_heads', type=int, default=12,
                        help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=3072,
                        help='Dimension of the feed-forward layer')
    parser.add_argument('--rope_theta', type=float, default=10000.0,
                        help='RoPE theta parameter')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--max_iters', type=int, default=100000,
                        help='Maximum number of training iterations')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Maximum learning rate')
    parser.add_argument('--min_learning_rate', type=float, default=3e-5,
                        help='Minimum learning rate')
    parser.add_argument('--warmup_iters', type=int, default=1000,
                        help='Number of warmup iterations')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='Weight decay for AdamW optimizer')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Beta1 for AdamW optimizer')
    parser.add_argument('--beta2', type=float, default=0.95,
                        help='Beta2 for AdamW optimizer')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping threshold')
    
    # Data paths
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data file (numpy memmap)')
    parser.add_argument('--val_data', type=str, required=True,
                        help='Path to validation data file (numpy memmap)')
    
    # Checkpointing and logging
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_interval', type=int, default=1000,
                        help='Save checkpoint every N iterations')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log training metrics every N iterations')
    parser.add_argument('--eval_interval', type=int, default=500,
                        help='Evaluate on validation set every N iterations')
    parser.add_argument('--eval_iters', type=int, default=100,
                        help='Number of iterations for validation evaluation')
    
    # Resume training
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    # Device and precision
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on (cuda/cpu)')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float16', 'bfloat16'],
                        help='Data type for model parameters')
    
    # Experiment tracking
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for the experiment (defaults to timestamp)')
    parser.add_argument('--experiment_log_dir', type=str, default='experiment_logs',
                        help='Directory to save experiment logs')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Weights & Biases project name (optional)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Weights & Biases run name (optional)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging even if installed')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def load_data(data_path, dtype=None):
    """Load data using memory-mapped numpy array for efficiency."""
    # Check if it's a .npy file
    if data_path.endswith('.npy'):
        # Load .npy file with memory mapping
        mmap_data = np.load(data_path, mmap_mode='r')
        print(f"  Loaded .npy file with dtype: {mmap_data.dtype}, shape: {mmap_data.shape}")
        return mmap_data
    else:
        # Fallback to original binary file loading
        if dtype is None:
            dtype = np.int32
        file_size = os.path.getsize(data_path)
        num_tokens = file_size // np.dtype(dtype).itemsize
        mmap_data = np.memmap(data_path, dtype=dtype, mode='r', shape=(num_tokens,))
        return mmap_data


def estimate_loss(model, data, batch_size, context_length, device, eval_iters):
    """Estimate loss on a dataset by averaging over multiple batches."""
    model.eval()
    losses = []
    
    with torch.no_grad():
        for _ in range(eval_iters):
            inputs, targets = get_batch(data, batch_size, context_length, device)
            logits = model(inputs)
            
            # Reshape for cross_entropy
            # logits: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
            # targets: (batch_size, seq_len) -> (batch_size * seq_len,)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            
            loss = cross_entropy(logits_flat, targets_flat)
            losses.append(loss.item())
    
    model.train()
    return np.mean(losses)


def get_parameter_count(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device and dtype
    device = torch.device(args.device)
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize experiment name
    if args.experiment_name is None:
        args.experiment_name = f"transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize experiment tracker
    use_wandb = HAS_WANDB and not args.no_wandb and args.wandb_project is not None
    experiment_tracker = ExperimentTracker(
        experiment_name=args.experiment_name,
        log_dir=args.experiment_log_dir,
        config=vars(args),
        use_wandb=use_wandb,
        wandb_project=args.wandb_project
    )
    
    # Log experiment start
    experiment_tracker.log_note(f"Starting training with {args.num_layers} layers, "
                                f"{args.d_model} hidden dimensions, and "
                                f"{args.num_heads} attention heads.")
    
    # Load data
    print(f"Loading training data from {args.train_data}...")
    train_data = load_data(args.train_data)
    print(f"Loading validation data from {args.val_data}...")
    val_data = load_data(args.val_data)
    print(f"Training data size: {len(train_data):,} tokens")
    print(f"Validation data size: {len(val_data):,} tokens")
    
    # Check vocab_size is adequate for the data
    max_train_token = int(np.max(train_data)) if len(train_data) > 0 else 0
    max_val_token = int(np.max(val_data)) if len(val_data) > 0 else 0
    max_token_id = max(max_train_token, max_val_token)
    
    print(f"Maximum token ID in data: {max_token_id}")
    
    if args.vocab_size <= max_token_id:
        print(f"ERROR: vocab_size ({args.vocab_size}) must be larger than the maximum token ID ({max_token_id})!")
        print(f"Automatically adjusting vocab_size to {max_token_id + 1}")
        args.vocab_size = max_token_id + 1
    
    # Initialize model
    print("\nInitializing model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=dtype
    )
    
    num_params = get_parameter_count(model)
    print(f"Model initialized with {num_params:,} trainable parameters")
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )
    
    # Resume from checkpoint if specified
    start_iter = 0
    if args.resume_from:
        print(f"\nResuming from checkpoint: {args.resume_from}")
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed from iteration {start_iter}")
    
    # Training loop
    print("\nStarting training...")
    model.train()
    
    # Save hyperparameters
    hparams_path = os.path.join(args.checkpoint_dir, 'hyperparameters.json')
    with open(hparams_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Timing
    start_time = time.time()
    
    for iter_num in range(start_iter, args.max_iters):
        # Get learning rate for this iteration
        lr = get_lr_cosine_schedule(
            iter_num, 
            args.learning_rate, 
            args.min_learning_rate,
            args.warmup_iters,
            args.max_iters
        )
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Sample batch
        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, device)
        
        # Forward pass
        logits = model(inputs)
        
        # Compute loss
        # Reshape for cross_entropy
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        loss = cross_entropy(logits_flat, targets_flat)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        gradient_clipping(model.parameters(), args.grad_clip)
        
        # Optimizer step
        optimizer.step()
        
        # Logging
        if iter_num % args.log_interval == 0:
            elapsed = time.time() - start_time
            iter_per_sec = (iter_num - start_iter + 1) / elapsed
            
            # Calculate gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            print(f"Iter {iter_num:6d} | Loss: {loss.item():.4f} | LR: {lr:.2e} | "
                  f"Grad Norm: {total_norm:.4f} | Iter/s: {iter_per_sec:.2f}")
            
            # Log metrics to experiment tracker
            experiment_tracker.log_metrics({
                'train_loss': loss.item(),
                'learning_rate': lr,
                'grad_norm': total_norm,
                'iter_per_sec': iter_per_sec
            }, step=iter_num)
        
        # Validation
        if iter_num % args.eval_interval == 0:
            val_loss = estimate_loss(
                model, val_data, args.batch_size, args.context_length, 
                device, args.eval_iters
            )
            
            # Calculate perplexity
            val_perplexity = math.exp(val_loss)
            
            print(f"Validation | Iter {iter_num:6d} | Loss: {val_loss:.4f} | "
                  f"Perplexity: {val_perplexity:.2f}")
            
            # Log validation metrics
            experiment_tracker.log_metrics({
                'val_loss': val_loss,
                'val_perplexity': val_perplexity
            }, step=iter_num)
        
        # Checkpointing
        if iter_num > 0 and iter_num % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, 
                f'checkpoint_iter_{iter_num:06d}.pt'
            )
            save_checkpoint(model, optimizer, iter_num, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
            # Log checkpoint info
            checkpoint_metrics = {
                'train_loss': loss.item(),
                'learning_rate': lr
            }
            # Add validation loss if available
            if iter_num % args.eval_interval == 0:
                checkpoint_metrics['val_loss'] = val_loss
                checkpoint_metrics['val_perplexity'] = val_perplexity
            
            experiment_tracker.save_checkpoint_info(checkpoint_path, iter_num, checkpoint_metrics)
            
            # Also save a "latest" checkpoint for easy resuming
            latest_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pt')
            save_checkpoint(model, optimizer, iter_num, latest_path)
    
    # Final checkpoint
    final_path = os.path.join(args.checkpoint_dir, 'checkpoint_final.pt')
    save_checkpoint(model, optimizer, args.max_iters, final_path)
    print(f"\nTraining completed! Final checkpoint saved to {final_path}")
    
    # Final validation
    final_val_loss = estimate_loss(
        model, val_data, args.batch_size, args.context_length, 
        device, args.eval_iters
    )
    final_val_perplexity = math.exp(final_val_loss)
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Final validation perplexity: {final_val_perplexity:.2f}")
    
    # Log final metrics
    experiment_tracker.log_metrics({
        'final_val_loss': final_val_loss,
        'final_val_perplexity': final_val_perplexity
    }, step=args.max_iters)
    
    # Close experiment tracker
    experiment_tracker.close()


if __name__ == '__main__':
    main()
