import math
from einops import rearrange
from jaxtyping import Bool, Float, Int
import torch
from torch import Tensor
import torch.nn as nn
from . import RoPE, softmax, Linear


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, " ... queries d_v"]:
        d_k = Q.shape[-1]
        K_ = K.transpose(-1, -2)
        scale = torch.sqrt(torch.tensor(d_k, dtype=Q.dtype, device=Q.device))
        logits = Q @ K_ / scale
        if mask is not None:
            mask = mask.to(logits.device)
            logits = torch.where(mask, logits, float("-inf"))
        return softmax(logits, -1) @ V


class MultiHeadSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.scaled_dot_product_attention = ScaledDotProductAttention()

    def forward(
        self,
        d_model: int, #Dimensionality of the Transformer block inputs. 
        num_heads: int, #Number of heads to use in multi-head self-attention.
        q_proj_weight: Float[Tensor, " d_k d_in"],
        k_proj_weight: Float[Tensor, " d_k d_in"],
        v_proj_weight: Float[Tensor, " d_v d_in"],
        o_proj_weight: Float[Tensor, " d_model d_v"],
        in_features: Float[Tensor, " ... sequence_length d_in"],
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        Q = in_features @ q_proj_weight.transpose(-1, -2)
        K = in_features @ k_proj_weight.transpose(-1, -2)
        V = in_features @ v_proj_weight.transpose(-1, -2)

        d_k = d_model // num_heads
        Q = rearrange(Q, "... seq (num_heads d_k) -> ... num_heads seq d_k", d_k=d_k)
        K = rearrange(K, "... seq (num_heads d_k) -> ... num_heads seq d_k", d_k=d_k)
        V = rearrange(V, "... seq (num_heads d_k) -> ... num_heads seq d_k", d_k=d_k)

        seq_len = in_features.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device)).bool()
        attn = self.scaled_dot_product_attention(Q, K, V, mask)
        attn = rearrange(attn, "... num_heads seq d_k -> ... seq (num_heads d_k)")
        return attn @ o_proj_weight.transpose(-1, -2)


class MultiHeadSelfAttentionWithRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float,
        max_seq_len: int,
        device=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.attn = ScaledDotProductAttention()
        self.rope = RoPE(theta, self.d_k, max_seq_len, device=device)
        self.q_proj = Linear(d_model, d_model, device=device)
        self.k_proj = Linear(d_model, d_model, device=device)
        self.v_proj = Linear(d_model, d_model, device=device)
        self.output_proj = Linear(d_model, d_model, device=device)

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        #print("mhsa x.shape: ", x.shape)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        #print("Q1.shape: ", Q.shape)

        Q = rearrange(Q, "... seq (num_heads d_k) -> ... num_heads seq d_k", d_k=self.d_k)
        K = rearrange(K, "... seq (num_heads d_k) -> ... num_heads seq d_k", d_k=self.d_k)
        V = rearrange(V, "... seq (num_heads d_k) -> ... num_heads seq d_k", d_k=self.d_k)

        #print("Q2.shape: ", Q.shape)
        # Q, K -> [batch_size num_heads seq_len d_k]
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        attn = self.attn(Q, K, V, mask)
        attn = rearrange(attn, "... num_heads seq d_k -> ... seq (num_heads d_k)")
        return self.output_proj(attn)
