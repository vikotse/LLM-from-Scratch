import torch
from torch import Tensor

from . import softmax


def top_p_sampling(logits, top_p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
    top_p_mask = cumulative_probs > top_p
    top_p_mask[:, 0] = False
    sorted_logits.masked_fill_(top_p_mask, -float("inf"))
    return logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)


@torch.no_grad()
def generate(
    model,
    input_ids: Tensor,
    stop_token_id: int,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Tensor:
    context_length = getattr(model, "context_length", None)
    if max_new_tokens <= 0:
        return input_ids

    batch_size = input_ids.size(0)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

    for _ in range(max_new_tokens):
        if context_length and input_ids.size(1) >= context_length:
            break
        if finished.all():
            break

        if context_length:
            context = input_ids[:, -context_length :]
        else:
            context = input_ids

        logits = model(context)
        next_token_logits = logits[:, -1, :] / temperature  # (batch, vocab_size)

        if top_p < 1.0:
            next_token_logits = top_p_sampling(next_token_logits, top_p)

        next_token_prob = softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(next_token_prob, num_samples=1)  # (batch, 1)

        # 批处理下按序列处理 stop_token：已结束的序列不再续写，该位置固定为 stop_token_id
        just_finished = next_token_id.squeeze(-1) == stop_token_id
        finished = finished | just_finished
        next_token_id = torch.where(
            finished.unsqueeze(-1),
            torch.full_like(next_token_id, stop_token_id),
            next_token_id,
        )

        input_ids = torch.cat([input_ids, next_token_id], dim=1)
    return input_ids