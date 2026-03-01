import os

import torch

from src.config import get_default_config
from src.optimizer import AdamW
from src.tokenizer import BPETokenizer
from src.transformer import TransformerLM
from src.utils import load_checkpoint

def main():
    cfg = get_default_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    best_ckpt_path = os.path.join(cfg.train.runs_dir, "ckpt.best.pt")
    if not os.path.exists(best_ckpt_path):
        print(f"model file not exists: {best_ckpt_path}")
        return

    tokenizer = BPETokenizer.from_files(
        vocab_path=cfg.vocab.vocab_path, 
        merges_path=cfg.vocab.merges_path,
        special_tokens=cfg.vocab.special_tokens,
    )
    stop_token_id = tokenizer.encode("<|endoftext|>")[0]

    model = TransformerLM(
        vocab_size=cfg.vocab.vocab_size,
        context_length=cfg.data.context_length,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layer,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        rope_theta=cfg.model.rope_theta,
    ).to(device)

    model = torch.compile(model)

    optimizer = AdamW(
        params=model.parameters(),
    )

    _ = load_checkpoint(best_ckpt_path, model, optimizer)

    model.eval()

    text = "Once upon a time"
    input_ids = torch.tensor([tokenizer.encode(text)], device=device)
    output_ids = model.generate(
        input_ids=input_ids,
        stop_token_id=stop_token_id,
        max_new_tokens=50,
        temperature=0.95,
        top_p=0.7,
    )
    print(tokenizer.decode(output_ids[0].tolist()))
    

if __name__ == "__main__":
    main()