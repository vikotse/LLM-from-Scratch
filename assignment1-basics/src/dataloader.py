import os

import numpy as np
import torch

class DataLoader:
    def __init__(self, data_path: str, np_dtype: str, device: str):
        self.dtype = np.dtype(np_dtype)
        nbytes = os.path.getsize(data_path)
        itemsize = self.dtype.itemsize
        n = nbytes // itemsize
        self.mm = np.memmap(data_path, mode="r", dtype=self.dtype, shape=(n,))
        self.dataset_tensor = torch.from_numpy(self.mm).to(device=device, dtype=torch.long)

    def get_batch(self, batch_size, context_length, device):
        total_samples = len(self.dataset_tensor) - context_length
        indices = torch.randint(0, total_samples, (batch_size,))

        batch_seq_in = torch.empty(
            (batch_size, context_length), dtype=torch.long, device=device
        )
        batch_seq_out = torch.empty_like(batch_seq_in)

        for i, st_idx in enumerate(indices.tolist()):
            st = int(st_idx)
            batch_seq_in[i] = self.dataset_tensor[st: st + context_length]
            batch_seq_out[i] = self.dataset_tensor[st + 1: st + 1 + context_length]

        return (
            batch_seq_in.to(device=device, dtype=torch.long),
            batch_seq_out.to(device=device, dtype=torch.long),
        )
