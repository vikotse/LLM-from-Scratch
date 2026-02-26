import math
import os
import random

from collections.abc import Iterable
from jaxtyping import Bool, Float, Int
import numpy.typing as npt
import torch
from torch import Tensor
from typing import IO, Any, BinaryIO

from .softmax import softmax


class TrieNode:
    def __init__(self):
        self.children = {}
        self.value = None
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: bytes, value: Any = None) -> None:
        node = self.root
        for char in word:
            char = bytes([char])
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.value = value
        node.is_end_of_word = True

    def search(self, word: bytes) -> Any:
        node = self.root
        for char in word:
            char = bytes([char])
            if char not in node.children:
                return None
            node = node.children[char]
        if node.is_end_of_word:
            return node.value
        return None

    def segment(self, word: bytes) -> list[bytes]:
        node = self.root
        result = []
        match_char = []
        for char in word:
            char = bytes([char])
            if char in node.children:
                node = node.children[char]
                match_char.append(char)
                if node.is_end_of_word:
                    result.append(b''.join(match_char))
                    match_char = []
                    node = self.root
            else:
                if match_char:
                    result.extend(match_char)
                    match_char = []
                result.append(char)
        if match_char:
            result.extend(match_char)
        return result


def cross_entropy_loss(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"]
):
    """
    before use cross_entropy_loss, conversion executed beforehand:
    inputs: [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
    targets: [batch_size, seq_len] -> [batch_size * seq_len]

    such as:
    run_cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1))
    """
    max_x = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - max_x
    log_sum_exp = shifted.logsumexp(dim=-1, keepdim=True)
    target_probs = inputs.gather(dim=-1, index=targets.unsqueeze(-1))
    log_probs = -(target_probs - max_x) + log_sum_exp
    loss = log_probs.mean()
    return loss


def get_batch_py(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
):
    sample = random.sample(range(0, len(dataset) - context_length), batch_size)
    batch_seq_in, batch_seq_out = [], []
    for st_idx in sample:
        seq_in = dataset[st_idx: st_idx + context_length]
        seq_out = dataset[st_idx + 1: st_idx + 1 + context_length]
        batch_seq_in.append(torch.from_numpy(seq_in))
        batch_seq_out.append(torch.from_numpy(seq_out))
    batch_seq_in = torch.stack(batch_seq_in).to(device)
    batch_seq_out = torch.stack(batch_seq_out).to(device)
    return batch_seq_in, batch_seq_out


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
):
    dataset_tensor = torch.from_numpy(dataset).to(device=device, dtype=torch.long)
    total_samples = len(dataset_tensor) - context_length
    indices = torch.randperm(total_samples)[:batch_size]
    
    batch_seq_in, batch_seq_out = [], []
    for st_idx in indices:
        seq_in = dataset_tensor[st_idx: st_idx + context_length]
        seq_out = dataset_tensor[st_idx + 1: st_idx + 1 + context_length]
        batch_seq_in.append(seq_in)
        batch_seq_out.append(seq_out)
    
    batch_seq_in = torch.stack(batch_seq_in).to(device=device, dtype=torch.long)
    batch_seq_out = torch.stack(batch_seq_out).to(device=device, dtype=torch.long)
    return batch_seq_in, batch_seq_out


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes]
):
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(ckpt, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
):
    ckpt = torch.load(src)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["iteration"]
