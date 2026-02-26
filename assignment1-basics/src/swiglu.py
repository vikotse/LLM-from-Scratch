import torch
import torch.nn as nn
from . import Linear

def SiLU(x):
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: str = None,
        dtype: torch.dtype = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.w1_weight = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.w2_weight = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        self.w3_weight = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.w1_weight)
        nn.init.trunc_normal_(self.w2_weight)
        nn.init.trunc_normal_(self.w3_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x @ self.w1_weight.T
        x2 = x @ self.w3_weight.T
        silu = SiLU(x1)
        return silu * x2 @ self.w2_weight.T


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: str = None,
        dtype: torch.dtype = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x2 = self.w3(x)
        silu = SiLU(x1)
        return self.w2(silu * x2)
