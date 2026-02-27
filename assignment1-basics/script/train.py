import math
import os
import torch
import tqdm

import numpy as np

from src.config import get_default_config
from src.dataloader import DataLoader
from src.optimizer import AdamW, lr_cosine_schedule
from src.tokenizer import BPETokenizer
from src.tracker import ExperimentTracker
from src.transformer import TransformerLM
from src.utils import cross_entropy_loss, save_checkpoint


def evaluate(cfg, model, data_loader, device):
    model.eval()
    losses = []
    for _ in range(cfg.data.eval_batches):
        x, y = data_loader.get_batch(
            batch_size=cfg.data.batch_size,
            context_length=cfg.data.context_length,
            device=device
        )
        logits = model(x)
        loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    cfg = get_default_config()

    torch.manual_seed(cfg.train.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    os.makedirs(cfg.train.runs_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.train.runs_dir, "ckpt.pt")
    best_ckpt_path = os.path.join(cfg.train.runs_dir, "ckpt.best.pt")

    # init logger
    logger = ExperimentTracker(cfg)

    # load dataset
    train_data_loader = DataLoader(
        data_path=cfg.data.train_bin_path,
        np_dtype=cfg.data.np_dtype,
    )

    eval_data_loader = DataLoader(
        data_path=cfg.data.valid_bin_path,
        np_dtype=cfg.data.np_dtype,
    )

    # init model
    model = TransformerLM(
        vocab_size=cfg.vocab.vocab_size,
        context_length=cfg.data.context_length,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layer,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        rope_theta=cfg.model.rope_theta,
    )

    # init optimizer
    optimizer = AdamW(
        params=model.parameters(),
        lr=cfg.optim.learning_rate,
        weight_decay=cfg.optim.weight_decay,
        betas=cfg.optim.betas,
        eps=cfg.optim.eps,
    )

    min_eval_loss = math.inf
    # training
    for it in tqdm.tqdm(range(cfg.train.max_step)):
        lr = lr_cosine_schedule(
            it=it,
            max_learning_rate=cfg.optim.max_learning_rate,
            min_learning_rate=cfg.optim.min_learning_rate,
            warmup_iters=cfg.optim.warmup_iters,
            cosine_cycle_iters=cfg.optim.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = train_data_loader.get_batch(
            batch_size=cfg.data.batch_size,
            context_length=cfg.data.context_length,
            device=device,
        )

        # forward propagation
        logits = model(x)
        loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), y.view(-1))

        # clear gradient to None
        optimizer.zero_grad(set_to_none=True)
        # back propagation
        loss.backward()

        # update parameter
        optimizer.step()

        if (it + 1) % cfg.train.train_log_step == 0:
            logger.log({"step": it + 1, "train/loss": loss.item()})

        if (it + 1) % cfg.train.eval_log_step == 0:
            eval_loss = evaluate(cfg, model, eval_data_loader, device)
            logger.log({"step": it + 1, "valid/loss": eval_loss})
            if eval_loss < min_eval_loss:
                min_eval_loss = eval_loss
                save_checkpoint(model, optimizer, it + 1, best_ckpt_path)

    # save model
    save_checkpoint(model, optimizer, cfg.train.max_step, ckpt_path)


if __name__ == "__main__":
    main()
    