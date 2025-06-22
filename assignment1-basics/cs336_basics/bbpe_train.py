import os
import regex as re
from typing import BinaryIO
from multiprocessing import Pool, cpu_count
from functools import partial
from collections import Counter
import time
from tqdm import tqdm

PAT_COMPILED = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))

def get_pair_stats(word_freqs: dict[tuple[int, ...], int]) -> Counter[tuple[int, int]]:
    pair_freqs = Counter()
    for word, freq in word_freqs.items():
        if len(word) < 2:
            continue
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_freqs[pair] += freq
    return pair_freqs

def pre_tokenize(
    args: tuple[str, dict[bytes, int], list[str], re.Pattern]
) -> list[tuple[int, ...]]:
    chunk, byte_to_token_id, special_tokens, delimiter_pattern_compiled = args
    
    special_tokens_bytes = {token.encode("utf-8") for token in special_tokens}
    
    words_list = []
    
    if delimiter_pattern_compiled:
        chunks = delimiter_pattern_compiled.split(chunk)
    else:
        chunks = [chunk]

    for sub_chunk in chunks:
        if not sub_chunk:
            continue
            
        sub_chunk_bytes = sub_chunk.encode("utf-8")
        if sub_chunk_bytes in special_tokens_bytes:
            token_id = byte_to_token_id[sub_chunk_bytes]
            words_list.append((token_id,))
        else:
            words = PAT_COMPILED.findall(sub_chunk)
            for word_str in words:
                if word_str:
                    byte_sequence = word_str.encode("utf-8")
                    id_sequence = tuple(byte_sequence)
                    words_list.append(id_sequence)

    return words_list

def calculate_optimal_chunk_count(file_size_gb: float, base_chunks: int = None) -> int:
    if base_chunks is None:
        base_chunks = cpu_count()
    
    if file_size_gb > 1.0:
        optimal_chunks = min(base_chunks * 4, int(file_size_gb * base_chunks * 2))
    else:
        optimal_chunks = base_chunks
    
    return max(optimal_chunks, 1)

def train_bbpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_chunks: int = None,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    if num_chunks is None:
        file_size = os.path.getsize(input_path)
        file_size_gb = file_size / (1024**3)
        num_chunks = calculate_optimal_chunk_count(file_size_gb)
        print(f"Auto-calculated optimal chunks: {num_chunks} for {file_size_gb:.2f}GB file")
    
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[len(vocab)] = token_bytes
    byte_to_token_id = {v: k for k, v in vocab.items()}
    
    delimiter_pattern_compiled = None
    if special_tokens:
        special_tokens_sorted = sorted(
            [t.encode("utf-8") for t in special_tokens], key=len, reverse=True
        )
        escaped_tokens = [re.escape(t.decode("utf-8")) for t in special_tokens_sorted]
        delimiter_pattern = "|".join(escaped_tokens)
        if delimiter_pattern:
            delimiter_pattern_compiled = re.compile(f"({delimiter_pattern})")

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_chunks, "<|endoftext|>".encode("utf-8")
        )

        chunk_args = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_bytes = f.read(end - start)
            chunk_args.append(
                (
                    chunk_bytes.decode("utf-8", errors="ignore"),
                    byte_to_token_id,
                    special_tokens,
                    delimiter_pattern_compiled,
                )
            )

    max_processes = min(cpu_count(), len(chunk_args))
    
    with Pool(processes=max_processes) as pool:
        start_time = time.time()
        print(f"Starting pre-tokenization with {max_processes} processes on {len(chunk_args)} chunks...")
        results = list(pool.map(pre_tokenize, chunk_args))
        end_time = time.time()
        print(f"Pre-tokenization time: {end_time - start_time:.2f} seconds")

    if not results:
        return {}, []

    print("Aggregating word frequencies...")
    aggregation_start = time.time()
    
    all_word_freqs = Counter()
    total_words_processed = 0
    
    for words_list in results:
        if words_list:
            all_word_freqs.update(words_list)
            total_words_processed += len(words_list)
    
    aggregation_end = time.time()
    print(f"Aggregation time: {aggregation_end - aggregation_start:.2f} seconds")
    print(f"Total words processed: {total_words_processed:,}")
    print(f"Unique word patterns: {len(all_word_freqs):,}")

    merges = []

    print(f"Initial vocab size: {len(vocab)}")
    print(f"Target vocab size: {vocab_size}")

    num_merges = vocab_size - len(vocab)
    pbar = tqdm(
        total=num_merges,
        desc="Performing BPE merges"
    )

    pair_freqs = get_pair_stats(all_word_freqs)
    start_time = time.time()

    for _ in range(num_merges):
        if not pair_freqs:
            break
        
        max_freq = max(pair_freqs.values())
        candidate_pairs = [p for p, f in pair_freqs.items() if f == max_freq]
        best_pair = max(candidate_pairs, key=lambda p: (vocab[p[0]], vocab[p[1]]))
        
        new_token_id = len(vocab)
        
        merged_token_bytes = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        vocab[new_token_id] = merged_token_bytes
        
        p1, p2 = best_pair
        newly_formed_word_freqs = Counter()
        words_to_remove = []

        for word, freq in all_word_freqs.items():
            if len(word) < 2:
                continue
            
            i = 0
            new_word = []
            merged = False
            while i < len(word):
                if i < len(word) - 1 and word[i] == p1 and word[i+1] == p2:
                    new_word.append(new_token_id)
                    merged = True
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            if merged:
                new_word_tuple = tuple(new_word)
                newly_formed_word_freqs[new_word_tuple] += freq
                words_to_remove.append(word)

                for i in range(len(word) - 1):
                    pair_freqs[(word[i], word[i+1])] -= freq
        
        for word in words_to_remove:
            del all_word_freqs[word]
        all_word_freqs.update(newly_formed_word_freqs)

        for new_word, freq in newly_formed_word_freqs.items():
            if len(new_word) < 2:
                continue
            for i in range(len(new_word) - 1):
                pair_freqs[(new_word[i], new_word[i+1])] = pair_freqs.get((new_word[i], new_word[i+1]), 0) + freq
        
        if best_pair in pair_freqs:
             del pair_freqs[best_pair]
        
        pbar.update(1)

    end_time = time.time()
    print(f"Merge time: {end_time - start_time:.2f} seconds")
    
    pbar.close()
    return vocab, merges
    
