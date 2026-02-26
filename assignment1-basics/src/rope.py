import torch
import torch.nn as nn
from einops import rearrange, einsum

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
            Construct the RoPE module and create buffers if needed.
            theta: float Θ value for the RoPE
            d_k: int dimension of query and key vectors
            max_seq_len: int Maximum sequence length that will be inputted
            device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.half_dim = d_k // 2

        freq_seq = torch.arange(self.half_dim, dtype=torch.float32, device=device)
        inv_freq = theta ** (-(2 * freq_seq) / d_k)
        pos = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        #print("pos.shape: ", pos.shape)
        #print("inv_freq.shape: ", inv_freq.shape)
        freqs = torch.outer(pos, inv_freq)
        #print("freqs.shape: ", freqs.shape)

        #freqs = einsum(pos, inv_freq, "i, j -> i j")

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)

        self.register_buffer('sin_cache', sin, persistent=False)
        self.register_buffer('cos_cache', cos, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        x: [batch_size (num_heads) seq_len d_k]
        """
        in_type = x.dtype
        x = x.to(torch.float32)
        #print("x.shape: ", x.shape)
        #print("token_positions.shape: ", token_positions.shape)

        sin = self.sin_cache[token_positions]
        cos = self.cos_cache[token_positions]
        #print("sin.shape: ", sin.shape)
        #print("cos.shape: ", cos.shape)

        x_pair = rearrange(x, " ... seq_len (pair two) ->  ... seq_len pair two", two=2)
        #print("x_pair.shape: ", x_pair.shape)

        # ... sequence_length group 1
        x0, x1 = torch.unbind(x_pair, dim=-1)
        #print("x0.shape: ", x0.shape)
        #print("x1.shape: ", x1.shape)

        # sin/cos shape: [batch_size, seq_len, d_k//2]
        # x0/x1 shape: [batch_size, num_heads, seq_len, d_k//2]
        # Need to add num_heads dimension and rearrange to match
        # Determine if x has num_heads dimension (check if x has 4 dimensions)
        if len(x.shape) == 4:
            # x shape: [batch_size, num_heads, seq_len, d_k]
            # sin/cos shape: [batch_size, seq_len, d_k//2]
            # Need to add num_heads dimension: [batch_size, 1, seq_len, d_k//2] then expand
            num_heads = x.shape[1]
            sin = sin.unsqueeze(1).expand(-1, num_heads, -1, -1)
            cos = cos.unsqueeze(1).expand(-1, num_heads, -1, -1)

        rot0 = x0 * cos - x1 * sin
        rot1 = x0 * sin + x1 * cos
        rot_pair = torch.stack([rot0, rot1], dim=-1)
        roped_x = rearrange(rot_pair, " ... seq_len pair two -> ... seq_len (pair two)")
        return roped_x.to(dtype=in_type)
