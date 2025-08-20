import os
import regex as re
import pickle
from typing import BinaryIO, Iterable, Iterator
from multiprocessing import Pool, cpu_count
from functools import partial
from collections import Counter
import time
from tqdm import tqdm

from cs336_basics.bbpe_train import pre_tokenize_and_count, find_chunk_boundaries, PAT_COMPILED

class BPE_Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, 
        and (optionally) a list of special tokens. This function should accept
        the following parameters:
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.token_to_id = {token: id for id, token in self.vocab.items()}
        # if special tokens are provided, add them to the vocab and token_to_id
        if special_tokens:
            for special_token in special_tokens:
                token_bytes = special_token.encode('utf-8')
                if token_bytes not in self.token_to_id:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token_bytes
                    self.token_to_id[token_bytes] = new_id

        self.bpe_ranks: dict[tuple[bytes, bytes], int] = {
            merge_pair: rank for rank, merge_pair in enumerate(self.merges)
        }

    @classmethod
    def from_files(cls, vocab_filepath: str | os.PathLike, merges_filepath: str | os.PathLike, special_tokens=None):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens
        """
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs
        """
        if not hasattr(self, "_special_delim"):
            if self.special_tokens:
                special_sorted = sorted(self.special_tokens, key=len, reverse=True)
                escaped = [re.escape(tok) for tok in special_sorted]
                self._special_delim = re.compile(f"({'|'.join(escaped)})")
            else:
                self._special_delim = None

        special_set = set(self.special_tokens or [])

        chunks = self._special_delim.split(text) if self._special_delim else [text]

        def apply_bpe(tokens: list[bytes]) -> list[bytes]:
            """Apply BPE merges inside a single word (list of byte tokens)."""
            if len(tokens) < 2:
                return tokens
            while True:
                best_rank = None
                best_idx = -1
                for i in range(len(tokens) - 1):
                    rank = self.bpe_ranks.get((tokens[i], tokens[i + 1]))
                    if rank is not None and (best_rank is None or rank < best_rank):
                        best_rank = rank
                        best_idx = i
                if best_rank is None:
                    break
                merged = tokens[best_idx] + tokens[best_idx + 1]
                tokens[best_idx : best_idx + 2] = [merged]
            return tokens

        output_ids: list[int] = []

        for chunk in chunks:
            if not chunk:
                continue
            if chunk in special_set:
                # Special token stays as single unit
                token_bytes = chunk.encode("utf-8")
                output_ids.append(self.token_to_id[token_bytes])
                continue

            for piece in PAT_COMPILED.findall(chunk):
                if not piece:
                    continue
                byte_seq = piece.encode("utf-8")
                word_tokens = [bytes([b]) for b in byte_seq]
                merged_tokens = apply_bpe(word_tokens)
                output_ids.extend(self.token_to_id[t] for t in merged_tokens)

        return output_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a 
        generator that lazily yields token IDs. This is required for memory-eﬀicient 
        tokenization of large files that we cannot directly load into memory.
        """
        for text in iterable:
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """
        To decode a sequence of integer token IDs back to raw text, we can simply look up each ID's corresponding
        entries in the vocabulary (a byte sequence), concatenate them together, and then decode the bytes to a
        Unicode string. Note that input IDs are not guaranteed to map to valid Unicode strings (since a user
        could input any sequence of integer IDs). In the case that the input token IDs do not produce a valid
        Unicode string, you should replace the malformed bytes with the oﬀicial Unicode replacement character
        U+FFFD.3 The errors argument of bytes.decode controls how Unicode decoding errors are handled, and
        using errors='replace' will automatically replace malformed data with the replacement marker
        """
        text = b''.join(self.vocab[id] for id in ids)
        return text.decode('utf-8', errors='replace')
        