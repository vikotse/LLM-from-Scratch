import os
import numpy as np

from .utils import get_batch

class DataLoader:
    def __init__(self, data_path: str, np_dtype: str):
        self.dtype = np.dtype(np_dtype)
        nbytes = os.path.getsize(data_path)
        itemsize = self.dtype.itemsize
        n = nbytes // itemsize
        self.mm = np.memmap(data_path, mode="r", dtype=self.dtype, shape=(n,))

    def get_batch(self, batch_size, context_length, device):
        return get_batch(
            dataset=self.mm,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
