from collections import defaultdict
import json

from src.config import get_default_config
from src.tokenizer import BPETokenizer


def main():
    cfg = get_default_config()

    vocab = defaultdict(bytes)
    merges = []
    tokenizer = BPETokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=cfg.vocab.special_tokens
    )

    vocab, merges = tokenizer.fast_train(
        file_path=cfg.data.train_data_path,
        vocab_size=cfg.vocab.vocab_size,
    )

    tokenizer.dump(cfg.vocab.vocab_path, cfg.vocab.merges_path)


if __name__ == "__main__":
    main()