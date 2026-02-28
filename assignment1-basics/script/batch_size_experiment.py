import gc
import torch

from src.config import get_default_config
from script import train


def run_training(batch_size):
    cfg = get_default_config()

    cfg.train.batch_size = batch_size

    def _get_default_config():
        return cfg

    train.get_default_config = _get_default_config
    train.main()
    print(f"finish training batch_size={batch_size}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    batch_sizes = [8, 16, 32, 64]
    for batch_size in batch_sizes:
        run_training(batch_size=batch_size)
