import os
import regex as re
from typing import BinaryIO
from multiprocessing import Pool, cpu_count
from functools import partial
from collections import Counter, defaultdict
import time
from tqdm import tqdm
import heapq

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

# def pre_tokenize(
#     args: tuple[str, dict[bytes, int], list[str], re.Pattern]
# ) -> list[tuple[int, ...]]:
#     chunk, byte_to_token_id, special_tokens, delimiter_pattern_compiled = args
    
#     special_tokens_bytes = {token.encode("utf-8") for token in special_tokens}
    
#     words_list = []
    
#     if delimiter_pattern_compiled:
#         chunks = delimiter_pattern_compiled.split(chunk)
#     else:
#         chunks = [chunk]

#     for sub_chunk in chunks:
#         if not sub_chunk:
#             continue
            
#         sub_chunk_bytes = sub_chunk.encode("utf-8")
#         if sub_chunk_bytes in special_tokens_bytes:
#             token_id = byte_to_token_id[sub_chunk_bytes]
#             words_list.append((token_id,))
#         else:
#             words = PAT_COMPILED.findall(sub_chunk)
#             for word_str in words:
#                 if word_str:
#                     byte_sequence = word_str.encode("utf-8")
#                     id_sequence = tuple(byte_sequence)
#                     words_list.append(id_sequence)

#     return words_list

def pre_tokenize_and_count(
    args: tuple[bytes, dict[str, int], re.Pattern]
) -> Counter:
    chunk_bytes, special_token_to_id, delimiter_pattern_compiled = args
    chunk = chunk_bytes.decode("utf-8", errors="ignore")
    special_tokens_set = set(special_token_to_id.keys())
    
    words_list = []
    
    if delimiter_pattern_compiled:
        sub_chunks = delimiter_pattern_compiled.split(chunk)
    else:
        sub_chunks = [chunk]

    for sub_chunk in sub_chunks:
        if not sub_chunk:
            continue
            
        if sub_chunk in special_tokens_set:
            token_id = special_token_to_id[sub_chunk]
            words_list.append((token_id,))
        else:
            # For non-special tokens, apply the standard GPT-2 pre-tokenization regex.
            for word_str in PAT_COMPILED.findall(sub_chunk):
                if word_str:
                    # Encode the resulting word strings to their byte representations.
                    byte_sequence = word_str.encode("utf-8")
                    id_sequence = tuple(byte_sequence)
                    words_list.append(id_sequence)

    return Counter(words_list)

def merge_counters(c1: Counter, c2: Counter) -> Counter:
    c1.update(c2)
    return c1

def train_bbpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_chunks: int = 8,
    num_processes: int = None,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    before_pretokenization_time = time.time()
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[len(vocab)] = token_bytes
    byte_to_token_id = {v: k for k, v in vocab.items()}
    
    special_token_to_id = {
        token: byte_to_token_id[token.encode("utf-8")] for token in special_tokens
    }

    delimiter_pattern_compiled = None
    if special_tokens:
        # Sort by length descending to handle overlapping tokens correctly
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
                    chunk_bytes,
                    special_token_to_id,
                    delimiter_pattern_compiled,
                )
            )

    processes_to_use = num_processes
    if processes_to_use is None:
        # Use a conservative default to prevent memory issues on machines with many cores.
        processes_to_use = min(cpu_count(), 8)
    
    processes_to_use = min(processes_to_use, len(chunk_args))

    before_pretokenization_time = time.time() - before_pretokenization_time
    print(f"Time taken before pretokenization: {before_pretokenization_time:.2f} seconds")
    
    all_word_freqs = Counter()
    start_time = time.time()
    with Pool(processes=processes_to_use) as pool:
        print(f"Starting pre-tokenization with {processes_to_use} processes on {len(chunk_args)} chunks...")
        
        # Use imap_unordered for memory efficiency. It returns an iterator.
        results_iterator = pool.imap_unordered(pre_tokenize_and_count, chunk_args)
        
        # Iterate through results as they complete and merge them one by one.
        for chunk_counter in tqdm(results_iterator, total=len(chunk_args), desc="Processing chunks"):
            all_word_freqs.update(chunk_counter)

    end_time = time.time()
    print(f"Pre-tokenization and initial counting time: {end_time - start_time:.2f} seconds")

    if not all_word_freqs:
        return {}, []

    total_words_processed = sum(all_word_freqs.values())
    
    print(f"Total words processed: {total_words_processed:,}")
    print(f"Unique word patterns: {len(all_word_freqs):,}")

    merges = []

    print(f"Initial vocab size: {len(vocab)}")
    print(f"Target vocab size: {vocab_size}")

    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, merges

    class Node:
        def __init__(self, value, word_freq):
            self.value = value
            self.word_freq = word_freq
            self.prev = None
            self.next = None

    class pq_item:
        def __init__(self, freq, id_pair, byte_pair):
            self.freq = freq
            self.id_pair = id_pair
            self.byte_pair = byte_pair 

        def __lt__(self, other):
            if self.freq != other.freq:
                return self.freq > other.freq
            return self.byte_pair > other.byte_pair

    pair_to_nodes = defaultdict(set)
    for word_tuple, count in all_word_freqs.items():
        if len(word_tuple) < 2:
            continue
        
        word_freq = {'count': count}
        
        head = Node(word_tuple[0], word_freq)
        prev_node = head
        for i in range(1, len(word_tuple)):
            current_node = Node(word_tuple[i], word_freq)
            prev_node.next = current_node
            current_node.prev = prev_node
            
            pair = (prev_node.value, current_node.value)
            pair_to_nodes[pair].add(prev_node)
            prev_node = current_node

    pair_freqs = Counter()
    for pair, nodes in pair_to_nodes.items():
        pair_freqs[pair] = sum(node.word_freq['count'] for node in nodes)
        
    pq = [
        pq_item(freq, p, (vocab[p[0]], vocab[p[1]]))
        for p, freq in pair_freqs.items()
    ]
    heapq.heapify(pq)

    pbar = tqdm(total=num_merges, desc="Performing BPE merges")
    start_time = time.time()

    for _ in range(num_merges):
        if not pq:
            break
            
        best_pair = None
        while pq:
            item = heapq.heappop(pq)
            if item.id_pair not in pair_freqs:
                continue
            if pair_freqs[item.id_pair] == item.freq:
                best_pair = item.id_pair
                break
        
        if best_pair is None:
            break

        p1, p2 = best_pair
        
        new_token_id = len(vocab)
        merged_token_bytes = vocab[p1] + vocab[p2]
        merges.append((vocab[p1], vocab[p2]))
        vocab[new_token_id] = merged_token_bytes
        nodes_to_process = list(pair_to_nodes[best_pair])
        for node1 in nodes_to_process:
            node2 = node1.next
            if node2 is None:
                continue
            word_freq = node1.word_freq['count']

            if node1.prev:
                left_node = node1.prev
                old_left_pair = (left_node.value, node1.value)
                pair_freqs[old_left_pair] -= word_freq
                heapq.heappush(pq, pq_item(pair_freqs[old_left_pair], old_left_pair, (vocab[old_left_pair[0]], vocab[old_left_pair[1]])))
                
                pair_to_nodes[old_left_pair].discard(left_node)
                new_left_pair = (left_node.value, new_token_id)
                pair_to_nodes[new_left_pair].add(left_node)
                pair_freqs[new_left_pair] += word_freq
                heapq.heappush(pq, pq_item(pair_freqs[new_left_pair], new_left_pair, (vocab[new_left_pair[0]], vocab[new_left_pair[1]])))

            if node2.next:
                right_node = node2.next
                old_right_pair = (node2.value, right_node.value)
                pair_freqs[old_right_pair] -= word_freq
                heapq.heappush(pq, pq_item(pair_freqs[old_right_pair], old_right_pair, (vocab[old_right_pair[0]], vocab[old_right_pair[1]])))
                
                new_right_pair = (new_token_id, right_node.value)
                pair_to_nodes[old_right_pair].discard(node2)
                pair_to_nodes[new_right_pair].add(node1)
                pair_freqs[new_right_pair] += word_freq
                heapq.heappush(pq, pq_item(pair_freqs[new_right_pair], new_right_pair, (vocab[new_right_pair[0]], vocab[new_right_pair[1]])))

            node1.value = new_token_id
            node1.next = node2.next
            if node2.next:
                node2.next.prev = node1

        del pair_freqs[best_pair]
        del pair_to_nodes[best_pair]
        
        pbar.update(1)
    
    end_time = time.time()
    print(f"Merge time: {end_time - start_time:.2f} seconds")
    
    pbar.close()
    return vocab, merges
    
