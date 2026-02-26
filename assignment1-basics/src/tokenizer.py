from collections import defaultdict, Counter
from collections.abc import Iterable, Iterator
from itertools import pairwise
import json
import math
from multiprocessing import Pool
import os
from sortedcontainers import SortedList
import time
from typing import IO, Any, BinaryIO

import regex as re

from .utils import Trie

_SPECIAL_TOKENS_TRIE = None

def _init_worker(trie) -> None:
    global _SPECIAL_TOKENS_TRIE
    _SPECIAL_TOKENS_TRIE = trie

def _chunk_tokenization(data_list: list[str]) -> None:
    special_tokens_trie = _SPECIAL_TOKENS_TRIE
    word_counter = Counter()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for data in data_list:
        for match in re.finditer(PAT, data):
            word = match.group().encode('utf-8')
            word_counter[tuple(special_tokens_trie.segment(word))] += 1
    return word_counter

_WORKER_TOKENIZER = None

def _init_encode_batch_worker(vocab_path, merges_path, special_tokens):
    """
    每个子进程中构造一个 BPETokenizer，只读共享 vocab/merges/special_tokens。
    """
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = BPETokenizer.from_files(
        vocab_path=vocab_path, 
        merges_path=merges_path,
        special_tokens=special_tokens,
    )

def _encode_line_batch_worker(lines: list[str]) -> list[list[int]]:
    """
    子进程中对一批行进行编码。
    输入: [line1, line2, ...]
    输出: [[ids_of_line1], [ids_of_line2], ...]（顺序保持不变）
    """
    results: list[list[int]] = []
    for line in lines:
        token_ids = _WORKER_TOKENIZER.encode(line)
        results.append(token_ids)
    return results


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes] = None,
        merges: list[tuple[bytes, bytes]] = None,
        special_tokens: list[str] = None
    ) -> None:
        self.__load(vocab, merges, special_tokens)

    @classmethod
    def from_files(
        cls,
        vocab_path: str,
        merges_path: str,
        special_tokens: list[str]
    ):
        #with open(vocab_path, "r") as vocab_file:
        #    vocab_serializable = json.load(vocab_file)
        #    vocab = {k: v.encode("utf-8") for k, v in vocab_serializable.items()}

        with open(vocab_path, "r", encoding="utf-8") as vocab_file:
            vocab_serializable = json.load(vocab_file)
            vocab = {int(k): v.encode("latin1") for k, v in vocab_serializable.items()}

        merges = []
        with open(merges_path, "rb") as merges_file:
            for line in merges_file:
                line = line.rstrip(b"\n")
                if not line:
                    continue
                token1, token2 = line.split(b" ", 1)
                merges.append((token1, token2))
        return cls(vocab, merges, special_tokens)

    def __load(
        self, 
        vocab: dict[int, bytes] = None,
        merges: list[tuple[bytes, bytes]] = None,
        special_tokens: list[str] = None
    ) -> None:
        self.vocab = vocab if vocab else defaultdict(bytes)
        self.vocab_trie = Trie()
        for id, token in self.vocab.items():
            self.vocab_trie.insert(token, id)

        self.merges = merges if merges else []

        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens.sort(key=len, reverse=True) # match special token delimiter from long to short
        self.special_tokens_trie = Trie()
        for token in self.special_tokens:
            token_id = self.vocab_trie.search(token.encode('utf-8'))
            self.special_tokens_trie.insert(token.encode('utf-8'), value=token_id)

    def __vocab_init(self, vocab_dict: dict[int, bytes], special_tokens: list[str]) -> None:
        for i in range(256):
            vocab_dict[i] = bytes([i])

        for token in special_tokens:
            vocab_dict[len(vocab_dict)] = token.encode('utf-8')

    def __pre_tokenization(self, file_path: str, special_tokens: list[str]) -> Counter:
        word_counter = Counter()
        with open(file_path, "r", encoding='utf-8') as f:
            SPECIAL_TOKEN_PAT = '|'.join([re.escape(token) for token in special_tokens]) 
            chunks = re.split(SPECIAL_TOKEN_PAT, f.read())

        num_cores = os.cpu_count() or 1
        chunks_per_core = math.ceil(len(chunks) / num_cores)
        tasks = [chunks[i * chunks_per_core: (i + 1) * chunks_per_core] for i in range(num_cores)]
        _init_worker(self.special_tokens_trie)

        t1 = time.time()
        with Pool(
                processes=num_cores,
                initializer=_init_worker,
                initargs=(self.special_tokens_trie,)
            ) as pool:
            word_sub_counters = pool.map(_chunk_tokenization, tasks)
        t2 = time.time()
        print(f"Tokenized chunks in {t2 - t1:.2f} seconds")

        word_counter = sum(word_sub_counters, Counter())
        t3 = time.time()
        print(f"Combined chunk counters in {t3 - t2:.2f} seconds")
        return word_counter

    def __find_most_common_bytes_pair(self, word_counter: Counter) -> tuple[bytes, bytes]:
        # compute pair frequencies
        t1 = time.time()
        token_pair_counter = Counter()
        for word_bytes_tuple, count in word_counter.items():
            for tok1, tok2 in pairwise(word_bytes_tuple):
                token_pair_counter[(tok1, tok2)] += count
        if not token_pair_counter:
            return None, None, None
        t2 = time.time()

        # find the most common pair
        max_count = 0
        pair = (b'', b'')
        for k, v in token_pair_counter.items():
            if v > max_count or v == max_count and k > pair:
                max_count = v
                pair = k
        t3 = time.time()
        return pair, t2 - t1, t3 - t2

    def __update_word_counter(self, word_counter: Counter, new_token: bytes) -> None:
        change_list = []
        for word_bytes_tuple, count in word_counter.items():
            # find if the new_token can be formed
            change_flag = False
            for tok1, tok2 in pairwise(word_bytes_tuple):
                if tok1 + tok2 == new_token:
                    change_flag = True
                    break
            if not change_flag:
                continue

            # convert to new tuple
            out_tokens = []
            for token in word_bytes_tuple:
                if out_tokens and out_tokens[-1] + token == new_token:
                    out_tokens[-1] = new_token
                else:
                    out_tokens.append(token)
            new_word_bytes_tuple = tuple(out_tokens)
            change_list.append((word_bytes_tuple, new_word_bytes_tuple, count))

        # apply the changes
        for old_word_bytes_tuple, new_word_bytes_tuple, count in change_list:
            del word_counter[old_word_bytes_tuple]
            word_counter[new_word_bytes_tuple] += count

    def __preprocess_token_counter(self, word_counter: Counter) -> dict[str, Any]:
        token_pair_counter = Counter()
        raw_token_id = 0
        id_to_token = defaultdict(bytes) # key: token id, value: token bytes
        prev_token_id = defaultdict(int) # key: token id, value: previous token id in the word
        next_token_id = defaultdict(int) # key: token id, value: next token id in the word
        token_ids_in_word = defaultdict(set) # key: token bytes, value: list of raw_token_id where this token appears
        token_word_count = defaultdict(int) # "st" in "newest", "newest" occurs 5 times -> position id ("st" of "newest") count 5. key: raw_token_id, value: count
        for word_bytes_tuple, count in word_counter.items():
            for i, token in enumerate(word_bytes_tuple):
                raw_token_id += 1
                id_to_token[raw_token_id] = token
                token_ids_in_word[token].add(raw_token_id)
                if i:
                    prev_token_id[raw_token_id] = raw_token_id - 1
                    next_token_id[raw_token_id - 1] = raw_token_id
                token_word_count[raw_token_id] = count
            for tok1, tok2 in pairwise(word_bytes_tuple):
                token_pair_counter[(tok1, tok2)] += count

        token_pair_freq_sl = SortedList()
        for token_pair, cnt in token_pair_counter.items():
            tok1, tok2 = token_pair
            token_pair_freq_sl.add((cnt, tok1, tok2))

        token_counter_result = {
            "id_to_token": id_to_token,
            "prev_token_id": prev_token_id,
            "next_token_id": next_token_id,
            "token_ids_in_word": token_ids_in_word,
            "token_word_count": token_word_count,
            "token_pair_counter": token_pair_counter,
            "token_pair_freq_sl": token_pair_freq_sl,
        }
        return token_counter_result

    def __compute_bpe_merges(
            self,
            token_counter_result: dict[str, Any],
            vocab_dict: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            vocab_size: int
        ) -> None:
        id_to_token = token_counter_result["id_to_token"]
        prev_token_id = token_counter_result["prev_token_id"]
        next_token_id = token_counter_result["next_token_id"]
        token_ids_in_word = token_counter_result["token_ids_in_word"]
        token_word_count = token_counter_result["token_word_count"]
        token_pair_counter = token_counter_result["token_pair_counter"]
        token_pair_freq_sl = token_counter_result["token_pair_freq_sl"]

        while len(vocab_dict) < vocab_size:
            if not token_pair_freq_sl:
                break

            _, merge_tok1, merge_tok2 = token_pair_freq_sl.pop()
            del token_pair_counter[(merge_tok1, merge_tok2)]

            new_token = merge_tok1 + merge_tok2
            vocab_dict[len(vocab_dict)] = new_token
            merges.append((merge_tok1, merge_tok2))

            # process all occurrences of (merge_tok1, merge_tok2)
            token1_remove_id_list = []
            token2_remove_id_list = []
            new_token_pair_counter = Counter()
            for token1_id in token_ids_in_word[merge_tok1]:
                token2_id = next_token_id[token1_id]
                # new_token occurs
                if token2_id and id_to_token[token2_id] == merge_tok2:
                    # merge new token pair with previous token: previous (s t) -> (e st)
                    if prev_token_id[token1_id]:
                        prev_token = id_to_token[prev_token_id[token1_id]]
                        new_token_pair_counter[(prev_token, new_token)] += token_word_count[token1_id]
                        new_token_pair_counter[(prev_token, merge_tok1)] -= token_word_count[token1_id]
                        next_token_id[prev_token_id[token1_id]] = token1_id
                    next_token_id[token1_id] = next_token_id[token2_id]

                    # merge new token pair with next token: next (s t) -> (st w)
                    if next_token_id[token2_id]:
                        next_token = id_to_token[next_token_id[token2_id]]
                        new_token_pair_counter[(new_token, next_token)] += token_word_count[token1_id]
                        new_token_pair_counter[(merge_tok2, next_token)] -= token_word_count[token1_id]
                        prev_token_id[next_token_id[token2_id]] = token1_id
                        next_token_id[token2_id] = 0
                        prev_token_id[token2_id] = 0

                    token_ids_in_word[new_token].add(token1_id)
                    id_to_token[token1_id] = new_token
                    token1_remove_id_list.append(token1_id)
                    token2_remove_id_list.append(token2_id)

            for token1_id, token2_id in zip(token1_remove_id_list, token2_remove_id_list):
                token_ids_in_word[merge_tok1].remove(token1_id)
                token_ids_in_word[merge_tok2].remove(token2_id)

            for (tok1, tok2), cnt in new_token_pair_counter.items():
                if (tok1, tok2) in token_pair_counter:
                    cur_cnt = token_pair_counter[(tok1, tok2)]
                    token_pair_freq_sl.remove((cur_cnt, tok1, tok2))
                    token_pair_counter[(tok1, tok2)] += cnt
                    cnt += cur_cnt
                if cnt > 0:
                    token_pair_freq_sl.add((cnt, tok1, tok2))
                    if (tok1, tok2) not in token_pair_counter:
                        token_pair_counter[(tok1, tok2)] = cnt
                else:
                    del token_pair_counter[(tok1, tok2)]

    def fast_train(self, file_path: str, vocab_size: int) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        vocab_dict = self.vocab
        merges = self.merges
        special_tokens = self.special_tokens

        t1 = time.time()
        self.__vocab_init(vocab_dict, special_tokens)
        t2 = time.time()
        print(f"Initialized vocab in {t2 - t1:.2f} seconds")
        word_counter = self.__pre_tokenization(file_path, special_tokens)
        t3 = time.time()
        print(f"Pre-tokenized data in {t3 - t2:.2f} seconds")

        # word_counter: word level counter, key: tuple of bytes representing pre-tokenized word
        # token_pair_counter: byte pair level counter, key: tuple of two bytes
        token_counter_result = self.__preprocess_token_counter(word_counter)
        t4 = time.time()
        print(f"Preprocessed token counter in {t4 - t3:.2f} seconds")
        self.__compute_bpe_merges(token_counter_result, vocab_dict, merges, vocab_size)
        t5 = time.time()
        print(f"Computed BPE merges in {t5 - t4:.2f} seconds")

        return vocab_dict, merges

    def slow_train(self, file_path: str, vocab_size: int) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        vocab_dict = self.vocab
        merges = self.merges
        special_tokens = self.special_tokens

        t1 = time.time()
        self.__vocab_init(vocab_dict, special_tokens)
        t2 = time.time()
        print(f"Initialized vocab in {t2 - t1:.2f} seconds")
        word_counter = self.__pre_tokenization(file_path, special_tokens)
        t3 = time.time()
        print(f"Pre-tokenized data in {t3 - t2:.2f} seconds")

        tts1 = tts2 = 0
        fts1 = fts2 = 0
        # word_counter: word level counter, key: tuple of bytes representing pre-tokenized word
        # token_pair_counter: byte pair level counter, key: tuple of two bytes
        t0 = time.time()
        while len(vocab_dict) < vocab_size:
            tt1 = time.time()
            res_bytes_pair, ft1, ft2 = self.__find_most_common_bytes_pair(word_counter)
            tt2 = time.time()
            tts1 += tt2 - tt1
            if not res_bytes_pair:
                break
            fts1 += ft1
            fts2 += ft2

            new_token = b''.join(res_bytes_pair)
            self.__update_word_counter(word_counter, new_token)
            tt3 = time.time()
            tts2 += tt3 - tt2

            vocab_dict[len(vocab_dict)] = new_token
            merges.append(res_bytes_pair)
        print(f"Finding most common pairs took {tts1:.2f} seconds")
        print(f"  - of which finding pair frequencies took {fts1:.2f} seconds")
        print(f"  - of which selecting most common pair took {fts2:.2f} seconds")
        print(f"Updating word counter took {tts2:.2f} seconds")
        t4 = time.time()
        print(f"Trained BPE in {t4 - t0:.2f} seconds")

        return vocab_dict, merges

    def __encode_text(self, text: str, result: list[int]) -> None:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for match in re.finditer(PAT, text):
            word = match.group().encode('utf-8')
            word_bytes_list = self.special_tokens_trie.segment(word)
            if len(word_bytes_list) == 1:
                token_id = self.vocab_trie.search(word_bytes_list[0])
                if token_id is None:
                    raise ValueError(f"Token not found in vocabulary: {word_bytes_list[0]}")
                result.append(token_id)
                continue

            id_to_token = defaultdict(bytes)
            token_ids_in_word = defaultdict(set)
            prev_token_id = defaultdict(int) # key: token id, value: previous token id in the word
            next_token_id = defaultdict(int) # key: token id, value: next token id in the word
            
            for i, char in enumerate(word_bytes_list):
                id_to_token[i + 1] = char
                token_ids_in_word[char].add(i + 1)
                if i:
                    prev_token_id[i + 1] = i
                    next_token_id[i] = i + 1

            for (token1, token2) in self.merges:
                if token1 not in token_ids_in_word or token2 not in token_ids_in_word:
                    continue

                new_token = token1 + token2
                token1_remove_id_list = []
                token2_remove_id_list = []
                remove_set = set()
                for id1 in token_ids_in_word[token1]:
                    id2 = next_token_id[id1]
                    if id1 in remove_set or id2 in remove_set:
                        continue
                    if id2 and id_to_token[id2] == token2:
                        # merge token1 and token2
                        id_to_token[id1] = new_token
                        token_ids_in_word[new_token].add(id1)
                        # update links
                        if next_token_id[id2]:
                            prev_token_id[next_token_id[id2]] = id1
                        next_token_id[id1] = next_token_id[id2]
                        prev_token_id[id2] = 0
                        next_token_id[id2] = 0

                        # remove old tokens
                        token1_remove_id_list.append(id1)
                        token2_remove_id_list.append(id2)
                        remove_set.update([id1, id2])

                for id1, id2 in zip(token1_remove_id_list, token2_remove_id_list):
                    token_ids_in_word[token1].remove(id1)
                    token_ids_in_word[token2].remove(id2)
            
            raw_id = 1
            while raw_id:
                token = id_to_token[raw_id]
                token_id = self.vocab_trie.search(token)
                if token_id is None:
                    raise ValueError(f"Token not found in vocabulary after BPE merging: {text}, {token}, {raw_id}, {len(word_bytes_list)}")
                result.append(token_id)
                raw_id = next_token_id[raw_id]

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            SPECIAL_TOKEN_PAT = '(' + '|'.join([re.escape(token) for token in self.special_tokens]) + ')'
            chunks = re.split(SPECIAL_TOKEN_PAT, text)
            chunks = [chunk for chunk in chunks if chunk]
        else:
            chunks = [text]

        result = []
        for chunk in chunks:
            special_token_id = self.special_tokens_trie.search(chunk.encode('utf-8'))
            if special_token_id:
                result.append(special_token_id)
                continue
            self.__encode_text(chunk, result)
        return result
        #raise NotImplementedError

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token_ids in self.encode(text):
                yield token_ids
        #raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        return b''.join([self.vocab[token_id] for token_id in ids]).decode('utf-8', errors='replace')
        #raise NotImplementedError

    def dump(self, vocab_path: str, merges_path: str) -> None:
        vocab_serializable = {
            #k: v.decode('utf-8', errors='replace') if isinstance(v, bytes) else v 
            k: v.decode('latin1', errors='replace') if isinstance(v, bytes) else v 
            for k, v in self.vocab.items()
        }
        #with open(vocab_path, "w") as vocab_file:
        vocab_dir_path = os.path.dirname(vocab_path)
        os.makedirs(vocab_dir_path, exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as vocab_file:
            json.dump(vocab_serializable, vocab_file, ensure_ascii=False, indent=4)

        merges_dir_path = os.path.dirname(merges_path)
        os.makedirs(merges_dir_path, exist_ok=True)
        with open(merges_path, "wb") as merges_file:
            for token1, token2 in self.merges:
                line = token1 + b' ' + token2 + b'\n'
                merges_file.write(line)
