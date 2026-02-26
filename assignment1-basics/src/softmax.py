from jaxtyping import Bool, Float, Int
import torch
from torch import Tensor

def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    in_max = torch.amax(in_features, dim=dim, keepdim=True)
    in_exp = (in_features - in_max).exp()
    in_deno = in_exp.sum(dim=dim, keepdim=True)
    return in_exp / in_deno
