#!/usr/bin/env python3
"""
Example script demonstrating text generation with a trained language model.
Shows practical usage of temperature scaling and top-p sampling.
"""

import torch
import numpy as np
from cs336_basics.transformer import TransformerLM, decode, load_checkpoint
from cs336_basics.bpe_tokenizer import BPE_Tokenizer
import pickle
import os
import argparse
import time


def load_tokenizer(tokenizer_path, merges_path=None):
    """Load the BPE tokenizer."""
    # If merges_path is provided, load vocab and merges separately
    if merges_path:
        try:
            with open(tokenizer_path, 'rb') as f:
                vocab = pickle.load(f)
            with open(merges_path, 'rb') as f:
                merges = pickle.load(f)
            
            special_tokens = ['<|endoftext|>']
            tokenizer = BPE_Tokenizer(vocab, merges, special_tokens)
            return tokenizer
        
        except Exception as e:
            print(f"Error loading separate vocab/merges files: {e}")
            raise
    
    # Otherwise, check if this is a pickled BPE tokenizer or old format
    try:
        with open(tokenizer_path, 'rb') as f:
            data = pickle.load(f)
        
        # If it's a BPE tokenizer saved as dict with vocab and merges
        if isinstance(data, dict) and 'vocab' in data and 'merges' in data:
            vocab = data['vocab']
            merges = data['merges']
            special_tokens = data.get('special_tokens', ['<|endoftext|>'])
            tokenizer = BPE_Tokenizer(vocab, merges, special_tokens)
            return tokenizer
        
        # If it's the old simple vocab format, create a simple mapping
        elif isinstance(data, (list, tuple)):
            vocab = data
            token_to_id = {token: idx for idx, token in enumerate(vocab)}
            id_to_token = {idx: token for idx, token in enumerate(vocab)}
            
            # Create a simple wrapper to mimic BPE tokenizer interface
            class SimpleTokenizer:
                def __init__(self, token_to_id, id_to_token):
                    self.token_to_id = token_to_id
                    self.id_to_token = id_to_token
                
                def encode(self, text):
                    tokens = []
                    words = text.split()
                    for word in words:
                        if word in self.token_to_id:
                            tokens.append(self.token_to_id[word])
                        else:
                            for char in word:
                                if char in self.token_to_id:
                                    tokens.append(self.token_to_id[char])
                                elif '<unk>' in self.token_to_id:
                                    tokens.append(self.token_to_id['<unk>'])
                    return tokens
                
                def decode(self, token_ids):
                    tokens = [str(self.id_to_token.get(tid, '<unk>')) for tid in token_ids]
                    text = ' '.join(tokens)
                    text = text.replace(' <|endoftext|>', '')
                    text = text.replace('<|endoftext|>', '')
                    return text.strip()
            
            return SimpleTokenizer(token_to_id, id_to_token)
    
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise


def decode_with_timing(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    endoftext_token_id: int = None,
    device: str = 'cuda',
    tokenizer = None
) -> tuple[torch.Tensor, dict]:
    """
    Generate text with detailed timing measurements for prefill and decode phases.
    
    Returns:
        tuple: (generated_tensor, timing_stats)
        timing_stats contains:
            - prefill_time: Time for processing the initial prompt
            - decode_time: Time for generating new tokens
            - prefill_speed: Tokens/second for prefill phase
            - decode_speed: Tokens/second for decode phase
            - total_time: Total generation time
    """
    model.eval()
    
    if prompt.device != device:
        prompt = prompt.to(device)
    
    prompt_length = prompt.shape[1]
    generated = prompt.clone()
    
    timing_stats = {
        'prefill_time': 0.0,
        'decode_time': 0.0,
        'prefill_speed': 0.0,
        'decode_speed': 0.0,
        'total_time': 0.0,
        'tokens_generated': 0
    }
    
    total_start_time = time.time()
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            step_start_time = time.time()
            
            # Forward pass
            logits = model(generated)
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature scaling
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-p sampling
            probs = torch.softmax(next_token_logits, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                for batch_idx in range(probs.shape[0]):
                    indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                    probs[batch_idx, indices_to_remove] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            
            # Track timing for prefill vs decode phases
            if i == 0:
                # First iteration is prefill (processing the original prompt)
                timing_stats['prefill_time'] = step_time
                timing_stats['prefill_speed'] = prompt_length / step_time if step_time > 0 else 0
            else:
                # Subsequent iterations are decode (generating new tokens)
                timing_stats['decode_time'] += step_time
            
            timing_stats['tokens_generated'] = i + 1
            
            
            # Check for end of text token
            if endoftext_token_id is not None:
                next_token_id = next_token.item()
                if next_token_id == endoftext_token_id:
                    print(f"Found <|endoftext|> token at step {i}, stopping generation")
                    break
            
            # Fallback: Check if decoded text contains endoftext patterns
            if tokenizer and i % 5 == 0:  # Check every 5 steps
                current_generated = generated[0, prompt_length:].tolist()
                current_text = tokenizer.decode(current_generated)
                if '<|endoftext|>' in current_text:
                    print(f"Found <|endoftext|> in text at step {i}, stopping generation")
                    break
    
    total_end_time = time.time()
    timing_stats['total_time'] = total_end_time - total_start_time
    
    # Calculate decode speed (tokens per second)
    if timing_stats['decode_time'] > 0 and timing_stats['tokens_generated'] > 1:
        decode_tokens = timing_stats['tokens_generated'] - 1  # Exclude prefill token
        timing_stats['decode_speed'] = decode_tokens / timing_stats['decode_time']
    
    model.train()
    return generated, timing_stats


def generate_text(
    model,
    prompt,
    tokenizer,
    max_new_tokens=100,
    temperature=1.0,
    top_p=0.95,
    device='cuda'
):
    """Generate text from a prompt using the model."""
    # Tokenize the prompt
    prompt_tokens = tokenizer.encode(prompt)
    if len(prompt_tokens) == 0:
        print("Warning: Could not tokenize prompt, using default tokens")
        prompt_tokens = [0]  # Use first token as fallback
    
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    # Get endoftext token ID based on BPE tokenizer implementation
    endoftext_id = None
    if hasattr(tokenizer, 'token_to_id'):
        # In BPE tokenizer, special tokens are stored as UTF-8 encoded bytes
        endoftext_bytes = '<|endoftext|>'.encode('utf-8')
        endoftext_id = tokenizer.token_to_id.get(endoftext_bytes, None)
    
    # Fallback: search through vocab directly
    if endoftext_id is None and hasattr(tokenizer, 'vocab'):
        endoftext_bytes = b'<|endoftext|>'
        for token_id, token_bytes in tokenizer.vocab.items():
            if token_bytes == endoftext_bytes:
                endoftext_id = token_id
                break
    
    print(f"Endoftext token ID: {endoftext_id}")
    
    # Verify endoftext token encoding works correctly
    if endoftext_id is not None:
        test_tokens = tokenizer.encode("<|endoftext|>")
        if endoftext_id in test_tokens:
            print(f"✓ Tokenizer correctly encodes <|endoftext|> as ID {endoftext_id}")
        else:
            print(f"⚠️ Warning: <|endoftext|> not encoded as expected ID {endoftext_id}")
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Prompt tokens ({len(prompt_tokens)}): {prompt_tokens[:20]}...")
    print(f"\nGenerating with temperature={temperature}, top_p={top_p}...")
    
    # Generate with timing
    output, timing_stats = decode_with_timing(
        model=model,
        prompt=prompt_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        endoftext_token_id=endoftext_id,
        device=device,
        tokenizer=tokenizer
    )
    
    # Extract generated tokens (excluding prompt)
    generated_tokens = output[0, len(prompt_tokens):].tolist()
    
    # Convert to text
    generated_text = tokenizer.decode(generated_tokens)
    
    print(f"\nGenerated {len(generated_tokens)} new tokens")
    print(f"Generated token IDs: {generated_tokens[:20]}{'...' if len(generated_tokens) > 20 else ''}")
    print(f"Generated text: '{generated_text}'")
    
    # Check if endoftext token was actually generated
    if endoftext_id is not None and endoftext_id in generated_tokens:
        endoftext_pos = generated_tokens.index(endoftext_id)
        print(f"✓ <|endoftext|> token (ID: {endoftext_id}) found at position {endoftext_pos}")
    elif endoftext_id is not None:
        if '<|endoftext|>' in generated_text:
            print(f"ℹ️  Stopping detected via text pattern (not exact token ID {endoftext_id})")
        else:
            print(f"ℹ️  No <|endoftext|> detected - generated {len(generated_tokens)} tokens")
    else:
        print("⚠️  Could not determine endoftext token ID")
    
    # Display timing statistics
    print(f"\nPerformance Statistics:")
    print(f"Total time: {timing_stats['total_time']:.3f}s")
    print(f"Prefill phase:")
    print(f"  Time: {timing_stats['prefill_time']:.3f}s") 
    print(f"  Tokens: {len(prompt_tokens)}")
    print(f"  Speed: {timing_stats['prefill_speed']:.1f} tokens/s")
    print(f"Decode phase:")
    print(f"  Time: {timing_stats['decode_time']:.3f}s")
    print(f"  Tokens: {timing_stats['tokens_generated'] - 1}")
    print(f"  Speed: {timing_stats['decode_speed']:.1f} tokens/s")
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(description='Generate text with a trained language model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, required=True,
                        help='Path to tokenizer file (BPE tokenizer pickle or vocab file)')
    parser.add_argument('--merges', type=str, default=None,
                        help='Path to merges file (for separate vocab/merges BPE files)')
    parser.add_argument('--prompt', type=str, default="Once upon a time",
                        help='Text prompt for generation')
    parser.add_argument('--max_tokens', type=int, default=384,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Temperature for sampling (0.1-2.0 typical)')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p threshold for nucleus sampling (0.9-0.95 typical)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on')
    
    args = parser.parse_args()
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    if args.merges:
        print(f"Loading merges from {args.merges}...")
    tokenizer = load_tokenizer(args.tokenizer, args.merges)
    
    # Get vocab size
    if hasattr(tokenizer, 'vocab'):
        vocab_size = len(tokenizer.vocab)
    elif hasattr(tokenizer, 'token_to_id'):
        vocab_size = len(tokenizer.token_to_id)
    else:
        vocab_size = 10000  # fallback
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Quick check if endoftext token exists in tokenizer
    endoftext_bytes = b'<|endoftext|>'
    endoftext_in_vocab = any(token_bytes == endoftext_bytes for token_bytes in tokenizer.vocab.values())
    
    if endoftext_in_vocab:
        endoftext_token_id = next(token_id for token_id, token_bytes in tokenizer.vocab.items() 
                                 if token_bytes == endoftext_bytes)
        print(f"Found <|endoftext|> in vocab at ID {endoftext_token_id}")
    else:
        print("WARNING: <|endoftext|> token not found in vocabulary!")
    
    # Initialize model (you need to match these parameters with your trained model)
    # These are example values - adjust based on your model configuration
    model_config = {
        'vocab_size': vocab_size,
        'context_length': 512,
        'd_model': 512,
        'num_layers': 4,
        'num_heads': 16,
        'd_ff': 1344,
        'rope_theta': 10000.0,
        'device': args.device
    }
    
    print(f"\nInitializing model with config:")
    for k, v in model_config.items():
        print(f"  {k}: {v}")
    
    model = TransformerLM(**model_config)
    
    # Check model vocab size vs endoftext token ID
    if endoftext_in_vocab:
        if endoftext_token_id >= model_config['vocab_size']:
            print(f"ERROR: <|endoftext|> token ID ({endoftext_token_id}) >= model vocab_size ({model_config['vocab_size']})")
            print(f"Model cannot generate this token! Consider adjusting model vocab_size.")
        else:
            print(f"✓ <|endoftext|> token ID ({endoftext_token_id}) < model vocab_size ({model_config['vocab_size']})")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    # Create a dummy optimizer for loading (won't be used for generation)
    optimizer = torch.optim.Adam(model.parameters())
    iteration = load_checkpoint(args.checkpoint, model, optimizer)
    print(f"Loaded checkpoint from iteration {iteration}")
    
    # Generate text with different settings
    print("\n" + "="*60)
    print("GENERATION EXAMPLES")
    print("="*60)
    
    # Example 1: User specified settings
    generate_text(
        model,
        args.prompt,
        tokenizer,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device
    )
    
    # Example 2: Conservative generation
    print("\n" + "-"*60)
    print("Conservative generation (low temperature)")
    generate_text(
        model,
        args.prompt,
        tokenizer,
        max_new_tokens=args.max_tokens,
        temperature=0.5,
        top_p=0.9,
        device=args.device
    )
    
    # Example 3: Creative generation
    print("\n" + "-"*60)
    print("Creative generation (high temperature)")
    generate_text(
        model,
        args.prompt,
        tokenizer,
        max_new_tokens=args.max_tokens,
        temperature=1.2,
        top_p=0.98,
        device=args.device
    )
    
    # Example 4: Greedy decoding
    print("\n" + "-"*60)
    print("Greedy decoding (temperature=0.1)")
    generate_text(
        model,
        args.prompt,
        tokenizer,
        max_new_tokens=args.max_tokens,
        temperature=0.1,
        top_p=1.0,
        device=args.device
    )


if __name__ == "__main__":
    # If running without arguments, show example usage
    import sys
    if len(sys.argv) == 1:
        print("Example usage:")
        print("python generate_text_example.py --checkpoint checkpoints/checkpoint_final.pt --tokenizer cs336_basics/tinystories_vocab.pkl --prompt 'The little girl'")
        print("\nFor BPE tokenizer with separate vocab and merges files:")
        print("python generate_text_example.py --checkpoint checkpoints/checkpoint_final.pt --tokenizer cs336_basics/tinystories_vocab.pkl --merges cs336_basics/tinystories_merges.pkl --prompt 'The little girl'")
        print("\nFor combined BPE tokenizer file:")
        print("python generate_text_example.py --checkpoint checkpoints/checkpoint_final.pt --tokenizer trained_tokenizer.pkl --prompt 'The little girl'")
    else:
        main()
