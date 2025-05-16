import os
import regex as re
from typing import BinaryIO
from multiprocessing import Pool, cpu_count
from functools import partial
from collections import Counter
import time
from tqdm import tqdm

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_tokenize(chunk: str, special_tokens: list[str]) -> tuple[list[list[bytes]]]:
    """Pre-tokenize a chunk of text.

    Args:
        chunk (str): The text to tokenize.
        special_tokens (list[str]): List of special tokens to split on.
    returns:
        tuple[list[list[bytes]], dict[tuple[bytes, ...], int]]: A tuple containing:
            - list[list[bytes]]: a list of "words", where each "word" is a list of its constituent bytes.
                                 Special tokens are represented as a list containing a single bytes object.
    """
    
    escaped_tokens = [re.escape(token) for token in special_tokens]
    delimiter_pattern = "|".join(escaped_tokens)

    if delimiter_pattern:
        chunks = re.split(f"({delimiter_pattern})", chunk)
    else:
        chunks = [chunk]

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    processed_tokens_as_list_of_byte_sequences = []
    for sub_chunk in chunks:
        if sub_chunk in special_tokens:
            processed_tokens_as_list_of_byte_sequences.append([sub_chunk.encode('utf-8')])
        else:
            words_from_pat = [match.group() for match in re.finditer(PAT, sub_chunk)]
            for word_str in words_from_pat:
                byte_sequence = [bytes([b]) for b in word_str.encode('utf-8')]
                processed_tokens_as_list_of_byte_sequences.append(byte_sequence)

    return processed_tokens_as_list_of_byte_sequences

def cal_freq(all_tokens: list[list[bytes]], special_tokens: list[str]) -> tuple[Counter[tuple[bytes, bytes]], dict[tuple[bytes, bytes], list[tuple[int, int]]]]:
    pair_freq = Counter()
    pair_pos = {}
    special_tokens_set = set(special_tokens) 

    for i, token_byte_sequence in enumerate(all_tokens): 
        is_current_token_special = False
        if len(token_byte_sequence) == 1: 
            try:
                decoded_token_str = token_byte_sequence[0].decode('utf-8')
                if decoded_token_str in special_tokens_set:
                    is_current_token_special = True
            except UnicodeDecodeError: 
                pass 

        if is_current_token_special:
            continue 

        
        for j in range(len(token_byte_sequence) - 1):
            pair = (token_byte_sequence[j], token_byte_sequence[j+1])
            pair_freq[pair] += 1
            if pair not in pair_pos:
                pair_pos[pair] = []
            pair_pos[pair].append((i, j))
    return pair_freq, pair_pos

def merge_pair(best_pair: tuple[bytes, bytes], all_tokens: list[list[bytes]], pair_freq: Counter[tuple[bytes, bytes]], pair_pos: dict[tuple[bytes, bytes], list[tuple[int, int]]]) :
    merged_symbol = best_pair[0] + best_pair[1] # The actual merged byte(s) string
    pair_freq.pop(best_pair)
    active_word_pos = set()
    
    # Create progress bar for positions to merge
    positions_to_merge = pair_pos[best_pair]
    pbar = tqdm(
        total=len(positions_to_merge),
        desc=f"Merging {best_pair}",
        leave=False  # Don't leave the progress bar after completion
    )
    
    for i, j in positions_to_merge:
        if j > 0:
            # prev pair
            prev_pair = (all_tokens[i][j-1], best_pair[0])
            pair_freq[prev_pair] -= 1
            if pair_freq[prev_pair] == 0:
                del pair_freq[prev_pair]
            pair_freq[(all_tokens[i][j-1], merged_symbol)] += 1
            active_word_pos.add(i)
        if j + 2 < len(all_tokens[i]):
            # next_pair
            next_pair = (best_pair[1], all_tokens[i][j+2])
            pair_freq[next_pair] -= 1
            if pair_freq[next_pair] == 0:
                del pair_freq[next_pair]
            pair_freq[(merged_symbol, all_tokens[i][j+2])] += 1
            active_word_pos.add(i)
        pbar.update(1)
    
    pbar.close()

    # Create progress bar for updating tokens
    pbar = tqdm(
        total=len(active_word_pos),
        desc="Updating tokens",
        leave=False
    )
    
    for i in active_word_pos:
        for j in range(len(all_tokens[i])-1):
            pair = (all_tokens[i][j], all_tokens[i][j+1])
            if pair in pair_pos:
                pair_pos[pair].remove((i, j))
        new_tokens = []
        j = 0
        while j < len(all_tokens[i]):
            if j < len(all_tokens[i])-1 and (all_tokens[i][j], all_tokens[i][j+1]) == best_pair:
                new_tokens.append(merged_symbol)
                j += 2
            else:
                new_tokens.append(all_tokens[i][j])
                j += 1
        all_tokens[i] = new_tokens
        for j in range(len(all_tokens[i])-1):
            pair = (all_tokens[i][j], all_tokens[i][j+1])
            if pair not in pair_pos:
                pair_pos[pair] = []
            pair_pos[pair].append((i, j))
        pbar.update(1)
    
    pbar.close()
    pair_pos.pop(best_pair)
    return all_tokens



def train_bbpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_chunks: int = 4,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    special_tokens_set = set(special_tokens)

    # single process
    # with open(input_path, 'rb') as f:
    #     boundaries = find_chunk_boundaries(f, num_chunks, "<|endoftext|>".encode("utf-8"))

    #     for start, end in zip(boundaries[:-1], boundaries[1:]):
    #         f.seek(start)
    #         chunk = f.read(end - start).decode("utf-8", errors="ignore")

    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_chunks, "<|endoftext|>".encode("utf-8"))

        # print(f"boundaries=({boundaries})")

        with Pool(processes=min(cpu_count(), num_chunks)) as pool:
            start_time = time.time()
            chunk_args = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end-start).decode("utf-8", errors="ignore")
                chunk_args.append((chunk, special_tokens))

            results = list(pool.starmap(pre_tokenize, chunk_args))

            all_tokens = []

            for tokens in results:
                all_tokens.extend(tokens)

            end_time = time.time()
            print(f"Pre-tokenization time: {end_time - start_time} seconds")

    # print(f"Total number of tokens: {len(all_tokens)}")
    # print(f"Sample of tokens: {all_tokens[:5]}")

    # optimized merging step
    vocab = {} # token_id -> bytes
    merges = [] # merge_rules
    pair_pos = {} # pair -> list[tuple[int, int]]
    for i in range(256):
        vocab[len(vocab)] = bytes([i])

    for i, token in enumerate(special_tokens):
        vocab[len(vocab)] = token.encode('utf-8')



    print(f"Initial vocab size: {len(vocab)}")
    print(f"Target vocab size: {vocab_size}")
    print(f"vocab: {vocab}")
    print(f"Progress bar total: {vocab_size - len(vocab)}")

    pbar = tqdm(
        total=vocab_size - len(vocab),
        desc="Performing BPE merges"
    )

    start_time = time.time()

    pair_freq, pair_pos = cal_freq(all_tokens, special_tokens)
    end_time = time.time()
    print(f"Frequency calculation time: {end_time - start_time} seconds")

    # print(f"sample pair_freq: {list(pair_freq.items())[:5]}")
    # print(f"sample all_tokens: {all_tokens[:5]}")
    # print(f"sample max pair: {max(pair_freq.items(), key=lambda x: x[1])}")

    top_12 = sorted(pair_freq.items(), key=lambda x: x[1], reverse=True)[:12]
    print(f"top_12: {top_12}")

    start_time = time.time()
    step = 0
    while len(vocab) < vocab_size:
        
        if len(pair_freq) == 0:
            break
        max_freq = max(pair_freq.values())
        # Get all pairs with max_freq
        candidates = [pair for pair, freq in pair_freq.items() if freq == max_freq]
        # if step > 25 and step < 32:
        #     for candidate in candidates:
        #         print(f"candidate: {candidate}, freq: {pair_freq[candidate]}")

        # Compare first elements, if equal then compare second elements
        pair = max(candidates, key=lambda x: (x[0], x[1]))  # lexicographically greatest
        if pair[0] in special_tokens_set or pair[1] in special_tokens_set:
            continue
        new_pair = pair[0] + pair[1]
        print(f"merge pair: {pair}")
        all_tokens = merge_pair(pair, all_tokens, pair_freq, pair_pos)
        merges.append(pair)
        vocab[len(vocab)] = new_pair
        step += 1
        pbar.update(1)
    end_time = time.time()
    print(f"Merge time: {end_time - start_time} seconds")
    
    pbar.close()
    print(f"vocab: {vocab}")
    print(f"merges: {merges}")
    return vocab, merges
    
