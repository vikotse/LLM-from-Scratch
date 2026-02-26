import os
from multiprocessing import Pool
import tqdm
import numpy as np

from src.config import get_default_config
from src.tokenizer import _init_encode_batch_worker, _encode_line_batch_worker

def batched(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def tokenize_and_bin(cfg, data_path, bin_path):
    all_ids = []

    num_workers = os.cpu_count() or 1
    batch_size = 64

    with open(data_path, "r", encoding="utf-8") as f, \
         Pool(
             processes=num_workers,
             initializer=_init_encode_batch_worker,
             initargs=(cfg.vocab.vocab_path, cfg.vocab.merges_path, cfg.vocab.special_tokens),
         ) as pool:

        line_batches = batched(f, batch_size)

        for batch_token_ids in tqdm.tqdm(pool.imap(_encode_line_batch_worker, line_batches)):
            for token_ids_per_line in batch_token_ids:
                all_ids.extend(token_ids_per_line)

    #tokenizer = BPETokenizer.from_files(
    #    vocab_path=cfg.vocab.vocab_path, 
    #    merges_path=cfg.vocab.merges_path,
    #    special_tokens=cfg.vocab.special_tokens,
    #)
    #with open(cfg.data.train_data_path) as f:
    #    for _id in tqdm.tqdm(tokenizer.encode_iterable(f)):
    #        all_ids.append(_id)
    print(len(all_ids))

    all_tokens = np.array(all_ids, dtype=np.dtype(cfg.data.np_dtype))

    dir_path = os.path.dirname(bin_path)
    os.makedirs(dir_path, exist_ok=True)
    all_tokens.tofile(bin_path)

def main():
    cfg = get_default_config()

    tokenize_and_bin(cfg, cfg.data.train_data_path, cfg.data.train_bin_path)
    tokenize_and_bin(cfg, cfg.data.valid_data_path, cfg.data.valid_bin_path)

if __name__ == "__main__":
    main()