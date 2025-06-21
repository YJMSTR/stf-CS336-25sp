import os
import regex as re
from typing import BinaryIO
from multiprocessing import Pool, cpu_count
from functools import partial
from collections import Counter
import time
from tqdm import tqdm

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None
        self.prev = None
    
    def __repr__(self):
        return f"Node(value={self.value}, next={self.next}, prev={self.prev})"
    
class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def append(self, value):
        new_node = Node(value)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node
    def __repr__(self):
        nodes = []
        cur = self.head
        while cur:
            nodes.append(cur.value)
            cur = cur.next
        nodes.append(None)
        return " -> ".join(str(node) for node in nodes)
    
    @staticmethod
    def from_list(values: list[int]) -> "DoublyLinkedList":
        dll = DoublyLinkedList()
        for value in values:
            dll.append(value)
        return dll
    

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

def pre_tokenize(chunk: str, special_tokens: list[str]) -> tuple[list[DoublyLinkedList], dict[int, bytes]]:
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

    processed_tokens_as_dll_list = []
    for sub_chunk in chunks:
        sub_chunk_bytes = sub_chunk.encode('utf-8')
        if sub_chunk_bytes in special_tokens_set:
            token_id = byte_to_token_id[sub_chunk_bytes]
            dll = DoublyLinkedList.from_list([token_id])
            processed_tokens_as_dll_list.append(dll)
        else:
            words_from_pat = [match.group() for match in re.finditer(PAT, sub_chunk)]
            for word_str in words_from_pat:
                byte_sequence = word_str.encode('utf-8')
                id_sequence = [b for b in byte_sequence]
                dll = DoublyLinkedList.from_list(id_sequence)
                processed_tokens_as_dll_list.append(dll)

    return processed_tokens_as_dll_list, vocab

def cal_freq(all_tokens_as_dll: list[DoublyLinkedList], special_token_ids: set[int]) -> tuple[Counter[tuple[int, int]], dict[tuple[int, int], set[Node]]]:
    pair_freq = Counter()
    pair_pos: dict[tuple[int, int], set[Node]] = {}

    for dll in all_tokens_as_dll:
        # Skip if it's a special token (linked list has only one node and it's a special token)
        if dll.head and dll.head.next is None and dll.head.value in special_token_ids:
            continue

        current_node = dll.head
        while current_node and current_node.next:
            pair = (current_node.value, current_node.next.value)
            pair_freq[pair] += 1
            if pair not in pair_pos:
                pair_pos[pair] = set()
            
            # 关键: 存储对Node对象的直接引用
            pair_pos[pair].add(current_node)
            
            current_node = current_node.next
            
    return pair_freq, pair_pos


def merge_pair(
    best_pair: tuple[int, int],
    pair_freq: Counter,
    pair_pos: dict,
    merged_token_id: int,
):
    nodes_to_merge = list(pair_pos[best_pair])

    for node1 in nodes_to_merge:
        node2 = node1.next
        
        if not (node1.value == best_pair[0] and node2 and node2.value == best_pair[1]):
            continue

        prev_node = node1.prev
        if prev_node:
            left_pair = (prev_node.value, node1.value)
            pair_freq[left_pair] -= 1
            if left_pair in pair_pos:
                pair_pos[left_pair].discard(prev_node)

        next_node = node2.next
        if next_node:
            right_pair = (node2.value, next_node.value)
            pair_freq[right_pair] -= 1
            if right_pair in pair_pos:
                pair_pos[right_pair].discard(node2)

        merged_node = Node(merged_token_id)
        merged_node.prev = prev_node
        merged_node.next = next_node
        
        if prev_node:
            prev_node.next = merged_node

        if next_node:
            next_node.prev = merged_node

        if prev_node:
            new_left_pair = (prev_node.value, merged_node.value)
            pair_freq[new_left_pair] = pair_freq.get(new_left_pair, 0) + 1
            if new_left_pair not in pair_pos:
                pair_pos[new_left_pair] = set()
            pair_pos[new_left_pair].add(prev_node)

        if next_node:
            new_right_pair = (merged_node.value, next_node.value)
            pair_freq[new_right_pair] = pair_freq.get(new_right_pair, 0) + 1
            if new_right_pair not in pair_pos:
                pair_pos[new_right_pair] = set()
            pair_pos[new_right_pair].add(merged_node)

    del pair_freq[best_pair]
    del pair_pos[best_pair]


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

    all_tokens: list[DoublyLinkedList] = []
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

    start_time = time.time()

    while len(vocab) < vocab_size:
        if not pair_freq:
            break
        
        max_freq = max(pair_freq.values())
        candidate_pairs = [p for p, f in pair_freq.items() if f == max_freq]
        pair = max(candidate_pairs, key=lambda p: (vocab[p[0]], vocab[p[1]]))
        
        new_token_id = len(vocab)
        
        merged_token_bytes = vocab[pair[0]] + vocab[pair[1]]
        merges.append((vocab[pair[0]], vocab[pair[1]]))
        vocab[new_token_id] = merged_token_bytes
        
        merge_pair(pair, pair_freq, pair_pos, new_token_id)

        pbar.update(1)

    end_time = time.time()
    print(f"Merge time: {end_time - start_time} seconds")
    
    pbar.close()
    return vocab, merges
    
