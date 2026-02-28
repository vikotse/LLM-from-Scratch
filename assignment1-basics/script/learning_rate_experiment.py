import gc
import torch

from src.config import get_default_config
from . import train


def run_training(lr):
    cfg = get_default_config()

    cfg.optim.learning_rate = lr
    cfg.optim.max_learning_rate = lr
    cfg.optim.min_learning_rate = lr / 10

    def _get_default_config():
        return cfg

    train.get_default_config = _get_default_config
    train.main()
    print(f"finish training lr={lr}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    lrs = [1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3]
    for lr in lrs:
        run_training(lr=lr)
