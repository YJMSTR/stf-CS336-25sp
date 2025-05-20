import os
import regex as re
import pickle
from typing import BinaryIO, Iterable, Iterator
from multiprocessing import Pool, cpu_count
from functools import partial
from collections import Counter
import time
from tqdm import tqdm

from cs336_basics.bbpe_train import pre_tokenize, find_chunk_boundaries 

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
        if special_tokens:
            for special_token in special_tokens:
                if special_token not in self.token_to_id:
                    self.vocab[len(self.vocab)] = special_token.encode('utf-8')
                    self.token_to_id[special_token] = len(self.vocab) - 1
        

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
        pre_tokenized = pre_tokenize(text, self.special_tokens or [])
        token_ids = []
        special_tokens_set = set(self.special_tokens or [])

        for merge in self.merges:
            for i, word in enumerate(pre_tokenized):
                word_str = b''.join(word).decode('utf-8')
                if special_tokens_set and word_str in special_tokens_set:
                    continue
                new_word = []
                j = 0
                while j < len(word):
                    if j+1 < len(word) and (word[j], word[j+1]) == merge:
                        new_word.append(word[j] + word[j+1])
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
                pre_tokenized[i] = new_word
        
        for word in pre_tokenized:
            word_str = b''.join(word).decode('utf-8')
            if special_tokens_set and word_str in special_tokens_set:
                token_ids.append(self.token_to_id[word_str.encode('utf-8')])
                print(f"Special token appended: {word_str}, ID: {self.token_to_id[word_str.encode('utf-8')]}")
                continue
            for token in word:
                if token in self.token_to_id:
                    token_ids.append(self.token_to_id[token])
                else:
                    raise ValueError(f"Token {token} not found in vocabulary")
        if len(token_ids) == 0:
            return []
        return token_ids

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
        