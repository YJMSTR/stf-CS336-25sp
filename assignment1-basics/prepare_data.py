#!/usr/bin/env python3
"""
Script to prepare TinyStories sample data for training.
Converts text file(s) to tokenized numpy arrays for training and validation.

Usage:
1. Single file mode (split into train/val):
   python prepare_data.py --input_file data.txt --train_ratio 0.9

2. Separate files mode:
   python prepare_data.py --train_input train.txt --val_input val.txt

Both modes support custom output paths and tokenizer files.
"""

import os
import numpy as np
import json
import pickle
import sys
import argparse

# Add the current directory to the path so we can import cs336_basics
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cs336_basics.bpe_tokenizer import BPE_Tokenizer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Prepare text data for Transformer training')
    
    # Input/Output files
    parser.add_argument('--input_file', type=str, default=None,
                        help='Single input text file to tokenize and split (for backward compatibility)')
    parser.add_argument('--train_input', type=str, default=None,
                        help='Training text file to tokenize')
    parser.add_argument('--val_input', type=str, default=None,
                        help='Validation text file to tokenize')
    parser.add_argument('--train_output', type=str, default="data/tinystories_sample_5M_train.npy",
                        help='Output file for training data')
    parser.add_argument('--val_output', type=str, default="data/tinystories_sample_5M_val.npy",
                        help='Output file for validation data')
    parser.add_argument('--output_dir', type=str, default="data",
                        help='Output directory for processed data')
    
    # Tokenizer files
    parser.add_argument('--vocab_file', type=str, default="cs336_basics/tinystories_vocab.pkl",
                        help='Path to vocabulary pickle file')
    parser.add_argument('--merges_file', type=str, default="cs336_basics/tinystories_merges.pkl",
                        help='Path to merges pickle file')
    
    # Data processing parameters
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Ratio of data to use for training (default: 0.9)')
    parser.add_argument('--chunk_size', type=int, default=100000,
                        help='Text chunk size for processing (default: 100000)')
    
    # Optional: Alternative tokenizer formats
    parser.add_argument('--vocab_json', type=str, default=None,
                        help='Path to vocabulary JSON file (GPT-2 style, optional)')
    parser.add_argument('--merges_txt', type=str, default=None,
                        help='Path to merges text file (GPT-2 style, optional)')
    
    return parser.parse_args()


def encode_text_file(input_file, tokenizer, output_file, chunk_size=100000):
    """
    Encode a text file to token IDs and save as numpy array.
    Process in chunks to avoid memory issues.
    """
    print(f"Encoding {input_file} to {output_file}...")
    
    token_ids = []
    total_tokens = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        text_chunk = ""
        for line in f:
            text_chunk += line
            
            # Process when chunk gets large enough
            if len(text_chunk) > chunk_size:
                chunk_tokens = tokenizer.encode(text_chunk)
                token_ids.extend(chunk_tokens)
                total_tokens += len(chunk_tokens)
                text_chunk = ""
                
                # Print progress
                if total_tokens % 100000 < len(chunk_tokens):
                    print(f"  Processed {total_tokens:,} tokens...")
        
        # Process remaining text
        if text_chunk:
            chunk_tokens = tokenizer.encode(text_chunk)
            token_ids.extend(chunk_tokens)
            total_tokens += len(chunk_tokens)
    
    print(f"  Total tokens: {total_tokens:,}")
    
    # Convert to numpy array
    # Use uint16 if vocabulary size allows, otherwise use int32
    max_token_id = max(token_ids) if token_ids else 0
    if max_token_id < 65536:
        dtype = np.uint16
        print(f"  Using uint16 (max token ID: {max_token_id})")
    else:
        dtype = np.int32
        print(f"  Using int32 (max token ID: {max_token_id})")
    
    # Save as numpy array
    token_array = np.array(token_ids, dtype=dtype)
    np.save(output_file, token_array)
    
    # Save file size info
    file_size = os.path.getsize(output_file)
    print(f"  Saved to {output_file} ({file_size / 1024 / 1024:.2f} MB)")
    
    return token_array


def split_data(token_array, train_ratio=0.9):
    """Split token array into train and validation sets."""
    n_tokens = len(token_array)
    n_train = int(n_tokens * train_ratio)
    
    train_tokens = token_array[:n_train]
    val_tokens = token_array[n_train:]
    
    print(f"\nData split:")
    print(f"  Training tokens: {len(train_tokens):,}")
    print(f"  Validation tokens: {len(val_tokens):,}")
    
    return train_tokens, val_tokens


def load_tokenizer(args):
    """Load tokenizer from specified files."""
    tokenizer = None
    
    # Try to load from pickle files first
    if args.vocab_file and args.merges_file:
        try:
            print(f"Attempting to load tokenizer from pickle files...")
            print(f"  Vocab file: {args.vocab_file}")
            print(f"  Merges file: {args.merges_file}")
            
            with open(args.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
            with open(args.merges_file, 'rb') as f:
                merges = pickle.load(f)
            
            # Include special tokens to ensure proper handling
            special_tokens = ['<|endoftext|>']
            tokenizer = BPE_Tokenizer(vocab, merges, special_tokens)
            print(f"  Loaded tokenizer with {len(vocab)} vocab size and special tokens: {special_tokens}")
            return tokenizer
            
        except Exception as e:
            print(f"  Failed to load from pickle files: {e}")
    
    # Try to load from JSON/text files if provided
    if args.vocab_json and args.merges_txt:
        try:
            print(f"Attempting to load tokenizer from JSON/text files...")
            print(f"  Vocab JSON: {args.vocab_json}")
            print(f"  Merges text: {args.merges_txt}")
            
            with open(args.vocab_json, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            # Convert JSON vocab to the format expected by BPE_Tokenizer
            vocab = {int(k): v.encode('utf-8') for k, v in vocab_data.items()}
            
            with open(args.merges_txt, 'r', encoding='utf-8') as f:
                merges_lines = f.read().strip().split('\n')
            
            # Parse merges from text format
            merges = []
            for line in merges_lines:
                if line and not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        merges.append((parts[0].encode('utf-8'), parts[1].encode('utf-8')))
            
            # Include special tokens to ensure proper handling
            special_tokens = ['<|endoftext|>']
            tokenizer = BPE_Tokenizer(vocab, merges, special_tokens)
            print(f"  Loaded tokenizer with {len(vocab)} vocab size and special tokens: {special_tokens}")
            return tokenizer
            
        except Exception as e:
            print(f"  Failed to load from JSON/text files: {e}")
    
    return None


def main():
    args = parse_args()
    
    # Validate input arguments
    if args.train_input and args.val_input:
        # Separate train/val files mode
        mode = "separate"
        if not os.path.exists(args.train_input):
            print(f"ERROR: Training input file does not exist: {args.train_input}")
            sys.exit(1)
        if not os.path.exists(args.val_input):
            print(f"ERROR: Validation input file does not exist: {args.val_input}")
            sys.exit(1)
        print(f"Using separate train/validation files:")
        print(f"  Training file: {args.train_input}")
        print(f"  Validation file: {args.val_input}")
    elif args.input_file:
        # Single file mode (backward compatibility)
        mode = "single"
        if not os.path.exists(args.input_file):
            print(f"ERROR: Input file does not exist: {args.input_file}")
            sys.exit(1)
        print(f"Using single file with train/val split:")
        print(f"  Input file: {args.input_file}")
        print(f"  Train/Val ratio: {args.train_ratio:.1f}/{1-args.train_ratio:.1f}")
    else:
        print("\nERROR: Must provide either:")
        print("  - Single file: --input_file <file>")
        print("  - Separate files: --train_input <train_file> --val_input <val_file>")
        sys.exit(1)
    
    # Load tokenizer
    tokenizer = load_tokenizer(args)
    
    if tokenizer is None:
        print("\nERROR: Could not load any tokenizer. Please provide valid tokenizer files.")
        print("\nSupported formats:")
        print("  - Pickle files: --vocab_file vocab.pkl --merges_file merges.pkl")
        print("  - JSON/text files: --vocab_json vocab.json --merges_txt merges.txt")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if mode == "separate":
        # Process train and validation files separately
        print(f"\nProcessing training file...")
        train_tokens = encode_text_file(args.train_input, tokenizer, args.train_output, args.chunk_size)
        
        print(f"\nProcessing validation file...")
        val_tokens = encode_text_file(args.val_input, tokenizer, args.val_output, args.chunk_size)
        
        print(f"\nData preparation complete!")
        print(f"  Training file: {args.train_input}")
        print(f"  Validation file: {args.val_input}")
        print(f"  Training data: {args.train_output} ({len(train_tokens):,} tokens)")
        print(f"  Validation data: {args.val_output} ({len(val_tokens):,} tokens)")
        
    else:
        # Single file mode - encode and split
        # Create temporary file for full tokenized data
        temp_tokens_file = os.path.join(args.output_dir, "temp_tokens.npy")
        
        print(f"\nEncoding text file: {args.input_file}")
        token_array = encode_text_file(args.input_file, tokenizer, temp_tokens_file, args.chunk_size)
        
        # Split data
        train_tokens, val_tokens = split_data(token_array, train_ratio=args.train_ratio)
        
        # Save train and validation data
        np.save(args.train_output, train_tokens)
        np.save(args.val_output, val_tokens)
        
        # Clean up temporary file
        if os.path.exists(temp_tokens_file):
            os.remove(temp_tokens_file)
        
        print(f"\nData preparation complete!")
        print(f"  Input file: {args.input_file}")
        print(f"  Training data: {args.train_output} ({len(train_tokens):,} tokens)")
        print(f"  Validation data: {args.val_output} ({len(val_tokens):,} tokens)")
        print(f"  Train/Val ratio: {args.train_ratio:.1f}/{1-args.train_ratio:.1f}")
    
    print(f"\nTo train with this data, run:")
    print(f"  python train.py \\")
    print(f"    --train_data {args.train_output} \\")
    print(f"    --val_data {args.val_output} \\")
    print(f"    --vocab_size {len(tokenizer.vocab)} \\")
    print(f"    --context_length 256 \\")
    print(f"    --d_model 512 \\")
    print(f"    --num_layers 4 \\")
    print(f"    --num_heads 16 \\")
    print(f"    --d_ff 1344 \\")
    print(f"    --rope_theta 10000 \\")
    print(f"    --batch_size 8 \\")
    print(f"    --max_iters 1000 \\")
    print(f"    --learning_rate 1e-3 \\")
    print(f"    --checkpoint_dir checkpoints/tinystories")


if __name__ == "__main__":
    main()
