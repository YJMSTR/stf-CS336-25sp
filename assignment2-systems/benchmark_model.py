#!/usr/bin/env python3
"""
Benchmarking script for CS336 Transformer model.

This script performs end-to-end benchmarking of forward and backward passes
in the Transformer model with configurable hyperparameters.
"""

import argparse
import time
import timeit
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from cs336_basics.transformer import TransformerLM
from cs336_basics.transformer import get_batch


MODEL_CONFIGS = {
    'small': {
        'd_model': 768,
        'd_ff': 3072,
        'num_layers': 12,
        'num_heads': 12
    },
    'medium': {
        'd_model': 1024,
        'd_ff': 4096,
        'num_layers': 24,
        'num_heads': 16
    },
    'large': {
        'd_model': 1280,
        'd_ff': 5120,
        'num_layers': 36,
        'num_heads': 20
    },
    'xl': {
        'd_model': 1600,
        'd_ff': 6400,
        'num_layers': 48,
        'num_heads': 25
    },
    '2.7B': {
        'd_model': 2560,
        'd_ff': 10240,
        'num_layers': 32,
        'num_heads': 32
    }
}

def create_random_dataset(context_length: int, vocab_size: int, num_tokens: int = 10000) -> np.ndarray:
    """Create a random dataset for benchmarking."""
    return np.random.randint(0, vocab_size, (num_tokens,))


def benchmark_model(
    model: nn.Module,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    num_warmup: int,
    num_steps: int,
    device: str,
    forward_only: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """
    Benchmark the model's forward and/or backward passes.
    
    Args:
        model: The Transformer model to benchmark
        batch_size: Batch size for the benchmark
        context_length: Context length for the benchmark
        vocab_size: Vocabulary size
        num_warmup: Number of warm-up steps
        num_steps: Number of steps to measure
        device: Device to run on ('cpu' or 'cuda')
        forward_only: If True, only benchmark forward pass; if False, benchmark both forward and backward
        optimizer: Optimizer instance (required for backward pass)
    
    Returns:
        Dictionary containing timing results
    """
    model.eval() if forward_only else model.train()
    
    dataset = create_random_dataset(context_length, vocab_size)
    x, y = get_batch(dataset, batch_size, context_length, device)
    
    # Warm-up phase
    print(f"Running {num_warmup} warm-up steps...")
    for _ in range(num_warmup):
        if forward_only:
            with torch.no_grad():
                _ = model(x)
        else:
            if optimizer is None:
                raise ValueError("Optimizer is required for backward pass benchmarking")
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.cross_entropy(output.view(-1, vocab_size), y.view(-1))
            loss.backward()
        
        if device == 'cuda':
            torch.cuda.synchronize()
    
    # Benchmarking phase
    print(f"Running {num_steps} benchmark steps...")
    forward_times = []
    backward_times = []
    
    for step in range(num_steps):
        if forward_only:
            # Forward pass only
            start_time = timeit.default_timer()
            with torch.no_grad():
                _ = model(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            forward_time = timeit.default_timer() - start_time
            forward_times.append(forward_time)
        else:
            # Forward pass
            optimizer.zero_grad()
            start_time = timeit.default_timer()
            output = model(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            forward_time = timeit.default_timer() - start_time
            
            # Backward pass
            loss = nn.functional.cross_entropy(output.view(-1, vocab_size), y.view(-1))
            start_time = timeit.default_timer()
            loss.backward()
            if device == 'cuda':
                torch.cuda.synchronize()
            backward_time = timeit.default_timer() - start_time
            
            forward_times.append(forward_time)
            backward_times.append(backward_time)
        
        if (step + 1) % 10 == 0:
            print(f"Completed {step + 1}/{num_steps} steps")
    
    # Calculate statistics
    results = {
        'device': device,
        'batch_size': batch_size,
        'context_length': context_length,
        'vocab_size': vocab_size,
        'num_warmup': num_warmup,
        'num_steps': num_steps,
        'forward_only': forward_only,
    }
    
    if forward_only:
        results['forward_times'] = forward_times
        results['forward_mean'] = sum(forward_times) / len(forward_times)
        results['forward_std'] = torch.tensor(forward_times).std().item()
        results['forward_min'] = min(forward_times)
        results['forward_max'] = max(forward_times)
    else:
        results['forward_times'] = forward_times
        results['backward_times'] = backward_times
        results['total_times'] = [f + b for f, b in zip(forward_times, backward_times)]
        
        results['forward_mean'] = sum(forward_times) / len(forward_times)
        results['forward_std'] = torch.tensor(forward_times).std().item()
        results['forward_min'] = min(forward_times)
        results['forward_max'] = max(forward_times)
        
        results['backward_mean'] = sum(backward_times) / len(backward_times)
        results['backward_std'] = torch.tensor(backward_times).std().item()
        results['backward_min'] = min(backward_times)
        results['backward_max'] = max(backward_times)
        
        results['total_mean'] = sum(results['total_times']) / len(results['total_times'])
        results['total_std'] = torch.tensor(results['total_times']).std().item()
        results['total_min'] = min(results['total_times'])
        results['total_max'] = max(results['total_times'])
    
    return results


def print_results(results: dict):
    """Print benchmarking results in a formatted way."""
    print("\n" + "="*60)
    print("BENCHMARKING RESULTS")
    print("="*60)
    print(f"Device: {results['device']}")
    print(f"Batch Size: {results['batch_size']}")
    print(f"Context Length: {results['context_length']}")
    print(f"Vocabulary Size: {results['vocab_size']}")
    print(f"Warm-up Steps: {results['num_warmup']}")
    print(f"Benchmark Steps: {results['num_steps']}")
    print(f"Forward Only: {results['forward_only']}")
    print()
    
    if results['forward_only']:
        print("FORWARD PASS TIMING:")
        print(f"  Mean: {results['forward_mean']*1000:.3f} ms")
        print(f"  Std:  {results['forward_std']*1000:.3f} ms")
        print(f"  Min:  {results['forward_min']*1000:.3f} ms")
        print(f"  Max:  {results['forward_max']*1000:.3f} ms")
    else:
        print("FORWARD PASS TIMING:")
        print(f"  Mean: {results['forward_mean']*1000:.3f} ms")
        print(f"  Std:  {results['forward_std']*1000:.3f} ms")
        print(f"  Min:  {results['forward_min']*1000:.3f} ms")
        print(f"  Max:  {results['forward_max']*1000:.3f} ms")
        print()
        
        print("BACKWARD PASS TIMING:")
        print(f"  Mean: {results['backward_mean']*1000:.3f} ms")
        print(f"  Std:  {results['backward_std']*1000:.3f} ms")
        print(f"  Min:  {results['backward_min']*1000:.3f} ms")
        print(f"  Max:  {results['backward_max']*1000:.3f} ms")
        print()
        
        print("TOTAL PASS TIMING:")
        print(f"  Mean: {results['total_mean']*1000:.3f} ms")
        print(f"  Std:  {results['total_std']*1000:.3f} ms")
        print(f"  Min:  {results['total_min']*1000:.3f} ms")
        print(f"  Max:  {results['total_max']*1000:.3f} ms")
    
    print("="*60)


def benchmark_multiple_configs(
    batch_size: int,
    context_length: int,
    vocab_size: int,
    num_warmup: int = 5,
    num_steps: int = 10,
    device: str = 'cuda',
    forward_only: bool = False,
    configs_to_test: list = None
) -> dict:
    """
    Benchmark multiple model configurations.
    
    Args:
        batch_size: Batch size for benchmarking
        context_length: Context length for benchmarking
        vocab_size: Vocabulary size
        num_warmup: Number of warm-up steps
        num_steps: Number of benchmark steps
        device: Device to run on
        forward_only: If True, only benchmark forward pass
        configs_to_test: List of config names to test (default: all configs)
    
    Returns:
        Dictionary containing results for all configurations
    """
    if configs_to_test is None:
        configs_to_test = list(MODEL_CONFIGS.keys())
    
    all_results = {}
    
    for config_name in configs_to_test:
        if config_name not in MODEL_CONFIGS:
            print(f"Warning: Unknown config '{config_name}', skipping...")
            continue
            
        config = MODEL_CONFIGS[config_name]
        print(f"\n{'='*60}")
        print(f"BENCHMARKING CONFIG: {config_name.upper()}")
        print(f"{'='*60}")
        print(f"d_model: {config['d_model']}")
        print(f"d_ff: {config['d_ff']}")
        print(f"num_layers: {config['num_layers']}")
        print(f"num_heads: {config['num_heads']}")
        print(f"{'='*60}")
        
        try:
            # Initialize model with current config
            model = TransformerLM(
                vocab_size=vocab_size,
                context_length=context_length,
                d_model=config['d_model'],
                num_layers=config['num_layers'],
                num_heads=config['num_heads'],
                d_ff=config['d_ff'],
                rope_theta=10000.0,
                device=device
            )
            
            model = model.to(device)
            
            # Initialize optimizer (only needed for backward pass)
            optimizer = None
            if not forward_only:
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
            # Run benchmark
            results = benchmark_model(
                model=model,
                batch_size=batch_size,
                context_length=context_length,
                vocab_size=vocab_size,
                num_warmup=num_warmup,
                num_steps=num_steps,
                device=device,
                forward_only=forward_only,
                optimizer=optimizer,
            )
            
            # Add config info to results
            results['config_name'] = config_name
            results['config'] = config
            
            all_results[config_name] = results
            
            # Print results for this config
            print_results(results)
            
            # Clean up
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error benchmarking config '{config_name}': {e}")
            all_results[config_name] = {'error': str(e)}
            continue
    
    return all_results


def print_summary_table(all_results: dict):
    """Print a summary table comparing all configurations."""
    print("\n" + "="*100)
    print("SUMMARY COMPARISON TABLE")
    print("="*100)
    
    # Header
    print(f"{'Config':<10} {'Params(M)':<10} {'Forward(ms)':<12} {'Backward(ms)':<13} {'Total(ms)':<10}")
    print("-" * 100)
    
    for config_name, results in all_results.items():
        if 'error' in results:
            print(f"{config_name:<10} {'ERROR':<10} {'N/A':<12} {'N/A':<13} {'N/A':<10}")
            continue
            
        config = results['config']
        num_params = results['config']['d_model'] * results['config']['num_layers'] * 4  # Rough estimate
        
        if results['forward_only']:
            forward_time = f"{results['forward_mean']*1000:.1f}"
            backward_time = "N/A"
            total_time = "N/A"
        else:
            forward_time = f"{results['forward_mean']*1000:.1f}"
            backward_time = f"{results['backward_mean']*1000:.1f}"
            total_time = f"{results['total_mean']*1000:.1f}"
        
        print(f"{config_name:<10} {num_params/1e6:<10.1f} {forward_time:<12} {backward_time:<13} {total_time:<10}")
    
    print("="*100)


def main():
    parser = argparse.ArgumentParser(description='Benchmark CS336 Transformer model')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for benchmarking')
    parser.add_argument('--context-length', type=int, default=512, help='Context length for benchmarking')
    parser.add_argument('--vocab-size', type=int, default=32000, help='Vocabulary size')
    parser.add_argument('--d-model', type=int, default=768, help='Model dimension')
    parser.add_argument('--num-layers', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--d-ff', type=int, default=3072, help='Feed-forward dimension')
    parser.add_argument('--rope-theta', type=float, default=10000.0, help='RoPE theta parameter')
    parser.add_argument('--num-warmup', type=int, default=5, help='Number of warm-up steps')
    parser.add_argument('--num-steps', type=int, default=10, help='Number of benchmark steps')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], 
                       help='Device to run on')
    parser.add_argument('--forward-only', action='store_true', help='Only benchmark forward pass')
    parser.add_argument('--all-configs', action='store_true', help='Benchmark all model configurations')
    parser.add_argument('--configs', nargs='+', choices=list(MODEL_CONFIGS.keys()), 
                       help='Specific configurations to benchmark (e.g., small medium large)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'
    
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Determine which configurations to benchmark
    if args.all_configs or args.configs:
        # Multi-config benchmark
        if args.all_configs:
            configs_to_test = None  # Will test all configs
            print("Benchmarking all model configurations...")
        else:
            configs_to_test = args.configs
            print(f"Benchmarking specific configurations: {', '.join(configs_to_test)}")
        
        # Run multi-config benchmark
        all_results = benchmark_multiple_configs(
            batch_size=args.batch_size,
            context_length=args.context_length,
            vocab_size=args.vocab_size,
            num_warmup=args.num_warmup,
            num_steps=args.num_steps,
            device=device,
            forward_only=args.forward_only,
            configs_to_test=configs_to_test,
        )
        
        # Print summary table
        print_summary_table(all_results)
        
    else:
        # Single config benchmark (original behavior)
        print("Initializing model...")
        model = TransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
            device=device
        )
        
        model = model.to(device)
        
        
        optimizer = None
        if not args.forward_only:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        print("Starting benchmark...")
        results = benchmark_model(
            model=model,
            batch_size=args.batch_size,
            context_length=args.context_length,
            vocab_size=args.vocab_size,
            num_warmup=args.num_warmup,
            num_steps=args.num_steps,
            device=device,
            forward_only=args.forward_only,
            optimizer=optimizer,
        )
        
        # Print results
        print_results(results)


if __name__ == '__main__':
    main()
