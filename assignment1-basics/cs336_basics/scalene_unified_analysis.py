#!/usr/bin/env python3

import sys
import os
sys.path.append(".")
import bbpe_train
import time

def unified_performance_analysis():

    
    print("start analysis")
    print("=" * 50)
    
    test_params = {
        "input_path": "../data/TinyStoriesV2-GPT4-train.txt",
        "vocab_size": 10000,
        "special_tokens": ["<|endoftext|>"],
        "num_chunks": 32,
    }
    
    print(f"test params:")
    for key, value in test_params.items():
        print(f"  {key}: {value}")
    
    print(f"\nstart training...")
    start_time = time.time()
    
    # train
    vocab, merges = bbpe_train.train_bbpe(**test_params)
    
    end_time = time.time()
    print(f"training done, time: {end_time - start_time:.2f} seconds")
    
    return vocab, merges

if __name__ == "__main__":
    if 'scalene' in sys.modules:
        print("running in scalene")
    else:
        print("please run with scalene")
        print("scalene --html --outfile analysis_report.html scalene_unified_analysis.py")
    
    vocab, merges = unified_performance_analysis()
    
    print(f"\nanalysis done!")
    print(f"vocab size: {len(vocab)}")
    print(f"merge rules: {len(merges)}") 