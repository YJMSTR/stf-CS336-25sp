import torch
from torch import nn
import einops

class Linear(nn.Module):
    def __init__(self, in_features, out_features,  device=None, dtype=None):
        """
        Construct a
        linear transformation module. This function should accept the following parameters:
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        # Initialize weight parameter with truncated normal distribution
        # Mean = 0, variance = 2/(in_features + out_features)
        std = (2 / (in_features + out_features)) ** 0.5

        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        
        # Fill with truncated normal distribution
        # Truncate at ±3 sigma
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input
        """
        return einops.einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")
