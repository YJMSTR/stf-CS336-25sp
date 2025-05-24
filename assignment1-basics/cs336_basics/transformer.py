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


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Construct an embedding module. This function should accept the following parameters:
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.

        Given a sequence of token IDs, the Transformer language model uses a token embedding layer to produce a sequence of vectors. Each embedding layer takes in a tensor of integers
        of shape (batch_size, sequence_length) and produces a sequence of vectors of shape (batch_size,
        sequence_length, d_model)
        """
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module. This function should accept the following parameters:

        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        x_square = x * x
        mean_square = einops.reduce(x_square, "batch_size seq_len d_model -> batch_size seq_len 1", "mean")
        mean_square = mean_square + self.eps
        mean_square = mean_square ** 0.5
        x = x / mean_square * self.weight
        x = x.to(in_dtype)
        return x

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        """
        Construct the SwiGLU module. You should set dff to approximately 8/3 * dmodel in your implementation, while ensuring that
        the dimensionality of the inner feed-forward layer is a multiple of 64 to make good use of your
        hardware. This function should accept the following parameters:
        
        d_model: int Hidden dimension of the model
        d_ff: int Dimension of the feed-forward layer
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = (d_ff + 63) // 64 * 64
        self.device = device
        self.dtype = dtype

        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU to the input
        """

        swish = self.w1.forward(x)
        swish = swish * torch.sigmoid(swish)
        gate = self.w3.forward(x)
        output = swish * gate
        output = self.w2.forward(output)
        return output

class RotatyPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.

        theta: float Theta value for RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))

        position_ids = torch.arange(max_seq_len, dtype=torch.float32, device=device)

        freqs = torch.outer(position_ids, freq) # theta_{i, k}
        cos_freqs = torch.cos(freqs)
        sin_freqs = torch.sin(freqs)

        self.register_buffer("cos_freqs", cos_freqs.to(device), persistent=False)
        self.register_buffer("sin_freqs", sin_freqs.to(device), persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. 
        Note that you should tolerate x with an arbitrary number of batch dimensions. You should assume
        that the token positions are a tensor of shape (..., seq_len) specifying the token positions of
        x along the sequence dimension.
        You should use the token positions to slice your (possibly precomputed) cos and sin tensors along
        the sequence dimension.
        """

        _, seq_len, d_k = x.shape
        assert d_k == self.d_k, "d_k must be equal to the d_k of the RoPE module"
        assert d_k % 2 == 0, "d_k must be even"

        x_pairs = einops.rearrange(x, "... seq_len (pairs two) -> ... seq_len pairs two", two = 2)

        x_real = x_pairs[..., 0]
        x_imag = x_pairs[..., 1]

        cos_freqs = self.cos_freqs[token_positions]
        sin_freqs = self.sin_freqs[token_positions]

        rotated_real = x_real * cos_freqs - x_imag * sin_freqs
        rotated_imag = x_real * sin_freqs + x_imag * cos_freqs

        x_pairs = torch.stack([rotated_real, rotated_imag], dim=-1)
        rotated_x = einops.rearrange(x_pairs, "... seq_len pairs two -> ... seq_len (pairs two)", two = 2)

        return rotated_x

def softmax_numerically_stable(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute the softmax of the input tensor in a numerically stable way.
    """
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_sum

