import os
import regex as re
from typing import BinaryIO
from multiprocessing import Pool, cpu_count
from functools import partial
from collections import Counter
import time
from tqdm import tqdm
import heapq

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

def pre_tokenize(chunk: str, special_tokens: list[str]) -> tuple[list[list[int]], dict[int, bytes]]:
    vocab = {i: bytes([i]) for i in range(256)}
    byte_to_token_id = {v: k for k, v in vocab.items()}

    for token in special_tokens:
        token_bytes = token.encode('utf-8')
        if token_bytes not in byte_to_token_id:
            token_id = len(vocab)
            vocab[token_id] = token_bytes
            byte_to_token_id[token_bytes] = token_id

    special_tokens_set = {token.encode('utf-8') for token in special_tokens}

    special_tokens_sorted = sorted([t.encode('utf-8') for t in special_tokens], key=len, reverse=True)
    
    escaped_tokens = [re.escape(token.decode('utf-8')) for token in special_tokens_sorted]
    delimiter_pattern = "|".join(escaped_tokens)

    if delimiter_pattern:
        chunks = re.split(f"({delimiter_pattern})", chunk)
    else:
        chunks = [chunk]

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    processed_tokens_as_list_of_ids = []
    for sub_chunk in chunks:
        sub_chunk_bytes = sub_chunk.encode('utf-8')
        if sub_chunk_bytes in special_tokens_set:
            token_id = byte_to_token_id[sub_chunk_bytes]
            processed_tokens_as_list_of_ids.append([token_id])
        else:
            words_from_pat = [match.group() for match in re.finditer(PAT, sub_chunk)]
            for word_str in words_from_pat:
                byte_sequence = word_str.encode('utf-8')
                id_sequence = [b for b in byte_sequence]
                processed_tokens_as_list_of_ids.append(id_sequence)

    return processed_tokens_as_list_of_ids, vocab

def cal_freq(all_tokens: list[list[int]], special_token_ids: set[int]) -> tuple[Counter[tuple[int, int]], dict[tuple[int, int], set[int]]]:
    pair_freq = Counter()
    pair_pos = {}

    for i, token_id_sequence in enumerate(all_tokens):
        if len(token_id_sequence) == 1 and token_id_sequence[0] in special_token_ids:
            continue

        for j in range(len(token_id_sequence) - 1):
            pair = (token_id_sequence[j], token_id_sequence[j+1])
            pair_freq[pair] += 1
            if pair not in pair_pos:
                pair_pos[pair] = set()
            
            packed_pos = (i << 32) | j
            pair_pos[pair].add(packed_pos)
            
    return pair_freq, pair_pos

class HeapItem:
    vocab: dict[int, bytes] = {}

    def __init__(self, freq: int, pair: tuple[int, int]):
        self.freq = freq
        self.pair = pair

    def __lt__(self, other: "HeapItem") -> bool:
        if self.freq != other.freq:
            return self.freq > other.freq
        
        p1_bytes = (HeapItem.vocab[self.pair[0]], HeapItem.vocab[self.pair[1]])
        p2_bytes = (HeapItem.vocab[other.pair[0]], HeapItem.vocab[other.pair[1]])
        return p1_bytes > p2_bytes

    def __repr__(self):
        return f"HeapItem(freq={self.freq}, pair={self.pair})"

def merge_pair(
    best_pair: tuple[int, int],
    all_tokens: list[list[int]],
    pair_freq: Counter[tuple[int, int]],
    pair_pos: dict[tuple[int, int], set[int]],
    pq: list,
    merged_token_id: int,
) -> None:
    affected_pair_types = set()

    indices_of_words_to_update = set(pos >> 32 for pos in pair_pos[best_pair])

    for i in indices_of_words_to_update:
        token_id_sequence = all_tokens[i]

        for j in range(len(token_id_sequence) - 1):
            pair = (token_id_sequence[j], token_id_sequence[j+1])
            if pair in pair_freq:
                pair_freq[pair] -= 1
                packed_pos = (i << 32) | j
                pair_pos[pair].remove(packed_pos)
                if pair_freq[pair] == 0:
                    del pair_freq[pair]
                    del pair_pos[pair]
                affected_pair_types.add(pair)

        new_token_id_sequence = []
        j = 0
        while j < len(token_id_sequence):
            if j < len(token_id_sequence) - 1 and (token_id_sequence[j], token_id_sequence[j+1]) == best_pair:
                new_token_id_sequence.append(merged_token_id)
                j += 2
            else:
                new_token_id_sequence.append(token_id_sequence[j])
                j += 1
        
        all_tokens[i] = new_token_id_sequence

        for j in range(len(new_token_id_sequence) - 1):
            pair = (new_token_id_sequence[j], new_token_id_sequence[j+1])
            if pair not in pair_pos:
                pair_pos[pair] = set()
            
            packed_pos = (i << 32) | j
            pair_pos[pair].add(packed_pos)
            pair_freq[pair] = pair_freq.get(pair, 0) + 1
            affected_pair_types.add(pair)
             
    if best_pair in pair_freq:
        del pair_freq[best_pair]
    if best_pair in pair_pos:
        del pair_pos[best_pair]
    if best_pair in affected_pair_types:
        affected_pair_types.remove(best_pair)

    for pair in affected_pair_types:
        if pair in pair_freq:
            heapq.heappush(pq, HeapItem(pair_freq[pair], pair))

def train_bbpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_chunks: int = 4,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_chunks, "<|endoftext|>".encode("utf-8"))
        
        chunk_args = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_bytes = f.read(end - start)
            chunk_args.append((chunk_bytes.decode("utf-8", errors="ignore"), special_tokens))

    with Pool(processes=min(cpu_count(), num_chunks)) as pool:
        start_time = time.time()
        results = list(pool.starmap(pre_tokenize, chunk_args))
        end_time = time.time()
        print(f"Pre-tokenization time: {end_time - start_time} seconds")

    all_tokens = []
    if not results:
        return {}, []

    vocab = results[0][1]
    
    for tokens, _ in results:
        all_tokens.extend(tokens)
    
    merges = []

    byte_to_token_id = {v: k for k, v in vocab.items()}
    special_token_ids = set()
    for st in special_tokens:
        st_bytes = st.encode('utf-8')
        if st_bytes in byte_to_token_id:
            special_token_ids.add(byte_to_token_id[st_bytes])

    print(f"Initial vocab size: {len(vocab)}")
    print(f"Target vocab size: {vocab_size}")
    print(f"Progress bar total: {vocab_size - len(vocab)}")

    pbar = tqdm(
        total=vocab_size - len(vocab),
        desc="Performing BPE merges"
    )

    start_time = time.time()
    pair_freq, pair_pos = cal_freq(all_tokens, special_token_ids)
    end_time = time.time()
    print(f"Frequency calculation time: {end_time - start_time} seconds")

    HeapItem.vocab = vocab

    pq = []
    for pair, freq in pair_freq.items():
        heapq.heappush(pq, HeapItem(freq, pair))

    start_time = time.time()

    while len(vocab) < vocab_size:
        if not pq:
            break
        top_item = None
        while pq:
            candidate = heapq.heappop(pq)
            if candidate.pair in pair_freq and pair_freq[candidate.pair] == candidate.freq:
                top_item = candidate
                break
        if top_item is None:
            break

        pair = top_item.pair
        
        new_token_id = len(vocab)
        
        merged_token_bytes = vocab[pair[0]] + vocab[pair[1]]
        merges.append((vocab[pair[0]], vocab[pair[1]]))
        vocab[new_token_id] = merged_token_bytes
        
        merge_pair(pair, all_tokens, pair_freq, pair_pos, pq, new_token_id)

        pbar.update(1)

    end_time = time.time()
    print(f"Merge time: {end_time - start_time} seconds")
    
    pbar.close()
    return vocab, merges
    
