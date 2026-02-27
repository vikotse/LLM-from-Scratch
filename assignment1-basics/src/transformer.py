import torch
from torch import Tensor
import torch.nn as nn
from . import RMSNorm, MultiHeadSelfAttentionWithRoPE, SwiGLUFFN, Embedding, Linear

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int, #Dimensionality of the Transformer block inputs.
        num_heads: int, #Number of heads to use in multi-head self-attention.
        d_ff: int, #Dimensionality of the position-wise feed-forward inner layer.
        max_seq_len,
        theta,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttentionWithRoPE(d_model, num_heads, theta, max_seq_len, device=device)
        self.ffn = SwiGLUFFN(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        token_positions = torch.arange(seq_len, device=x.device).repeat(batch_size, 1)
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        param_dtype = (
            dtype
            if (
                dtype is not None
                and torch.is_floating_point(torch.tensor([], dtype=dtype))
            )
            else torch.float32
        )
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=param_dtype)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=param_dtype,
            ))
        self.ln_final = RMSNorm(d_model, device=device, dtype=param_dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=param_dtype)

    def forward(self, x):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.ln_final(x))
        return logits
