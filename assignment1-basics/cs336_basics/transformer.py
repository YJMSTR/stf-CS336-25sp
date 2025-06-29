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

        seq_len, d_k = x.shape[-2:]
        assert d_k == self.d_k, f"d_k must be equal to the d_k of the RoPE module, got {d_k} and {self.d_k}"
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

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
    """
    Implement the scaled dot-product attention function. Your implementation should
    handle keys and queries of shape (batch_size, ..., seq_len, d_k) and values of shape
    (batch_size, ..., seq_len, d_v), where ... represents any number of other batch-like
    dimensions (if provided). The implementation should return an output with the shape (batch_size,
    ..., d_v). See section 3.3 for a discussion on batch-like dimensions.
    Your implementation should also support an optional user-provided boolean mask of shape (seq_len,
    seq_len). The attention probabilities of positions with a mask value of True should collectively sum
    to 1, and the attention probabilities of positions with a mask value of False should be zero.
    """
    batch_size = Q.shape[0]
    seq_len, d_k = Q.shape[-2:]
    d_v = V.shape[-1]

    assert Q.shape[-1] == K.shape[-1], "d_k must be the same for Q and K"
    assert K.shape[-2] == V.shape[-2], "seq_len must be the same for K and V"

    pre_softmax = einops.einsum(Q, K, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k")

    if mask is not None:
        pre_softmax = pre_softmax.masked_fill(~mask, float("-inf"))

    pre_softmax = pre_softmax / (d_k ** 0.5)

    softmax = softmax_numerically_stable(pre_softmax, dim=-1)
    # Verify that each column in softmax sums to 1
    softmax_sum = torch.sum(softmax, dim=-1)
    assert torch.allclose(softmax_sum, torch.ones_like(softmax_sum)), "Softmax probabilities do not sum to 1"

    return einops.einsum(softmax, V, "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float | None = None, max_seq_len: int | None = None, token_positions: torch.Tensor | None = None, device=None, dtype=None):
        """
        Casual Multihead Self-Attention module.

        d_model: int Hidden dimension of the model
        num_heads: int Number of attention heads
        theta: float | None = None RoPE parameter
        max_seq_len: int | None = None Maximum sequence length for RoPE
        token_positions: torch.Tensor | None = None Token positions for RoPE
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters

        As a stretch goal, try combining the key, query, and value projections 
        into a single weight matrix so you only need a single matrix multiply.
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.token_positions = token_positions
        self.d_k = self.d_v = d_model // num_heads

        if theta is not None:
            self.rope = RotatyPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)
        else:
            self.rope = None

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        # Apply RoPE to Q and K for each head
        if self.theta is not None:
            assert self.token_positions is not None, "token_positions must be provided if theta is not None"
            Q = self.rope.forward(Q, self.token_positions)
            K = self.rope.forward(K, self.token_positions)

        self_mask = torch.triu(torch.ones(Q.shape[0], Q.shape[2], K.shape[2], device=Q.device), diagonal=1).bool()
        self_mask = ~self_mask
        # print(self_mask)

        attention = scaled_dot_product_attention(Q, K, V, self_mask)

        attention = self.merge_heads(attention)

        return self.o_proj(attention)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension of the tensor into num_heads different dimensions.
        """
        return einops.rearrange(x, "batch_size seq_len (heads d_k) -> batch_size heads seq_len d_k", heads = self.num_heads)
    
    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge the last dimension of the tensor into num_heads different dimensions.
        """
        return einops.rearrange(x, "batch_size heads seq_len d_k -> batch_size seq_len (heads d_k)")
        
