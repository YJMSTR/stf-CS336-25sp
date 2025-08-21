import torch
from torch import nn
import einops
import numpy as np

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
        
        # Add num_heads dimension for broadcasting with x_real and x_imag
        # cos_freqs and sin_freqs have shape (..., seq_len, pairs)
        # x_real and x_imag have shape (..., num_heads, seq_len, pairs)
        # We need to insert a dimension at position 1 (for num_heads)
        cos_freqs = cos_freqs.unsqueeze(-3)  # (..., 1, seq_len, pairs)
        sin_freqs = sin_freqs.unsqueeze(-3)  # (..., 1, seq_len, pairs)

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

        # Create causal mask with correct dimensions for multi-head attention
        # Q.shape: [batch_size, num_heads, seq_len, d_k]
        batch_size, num_heads, seq_len = Q.shape[0], Q.shape[1], Q.shape[2]
        self_mask = torch.triu(torch.ones(batch_size, num_heads, seq_len, seq_len, device=Q.device), diagonal=1).bool()
        self_mask = ~self_mask

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


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int, device=None, dtype=None):
        """
        Pre-norm Transformer block with RoPE.
        
        d_model: int Hidden dimension of the model
        num_heads: int Number of attention heads  
        d_ff: int Dimension of the feed-forward layer
        theta: float RoPE parameter
        max_seq_len: int Maximum sequence length for RoPE
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        
        # Pre-norm layers for both sublayers
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        
        # Multi-head self-attention with RoPE 
        # Initialize with dummy token_positions, will be updated in forward
        dummy_positions = torch.zeros((1, max_seq_len), dtype=torch.long)
        self.attn = MultiheadSelfAttention(
            d_model, num_heads, theta, max_seq_len, 
            dummy_positions, device, dtype
        )
        
        # Feed-forward network (SwiGLU)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass of the transformer block implementing both sublayers:
        
        First sublayer: y = x + MultiHeadSelfAttention(RMSNorm(x))
        Second sublayer: z = y + FFN(RMSNorm(y))
        
        x: torch.Tensor of shape (batch_size, sequence_length, d_model)
        token_positions: torch.Tensor of shape (batch_size, sequence_length) with token positions for RoPE
        
        Returns: torch.Tensor of shape (batch_size, sequence_length, d_model)
        """
        # Generate default token positions if not provided
        if token_positions is None:
            batch_size, seq_len = x.shape[0], x.shape[1]
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Update attention's token positions
        self.attn.token_positions = token_positions
        
        # First sublayer: y = x + MultiHeadSelfAttention(RMSNorm(x))
        y = x + self.attn(self.ln1(x))
        
        # Second sublayer: z = y + FFN(RMSNorm(y))
        z = y + self.ffn(self.ln2(y))
        
        return z


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
        dtype=None
    ):
        """
        Transformer Language Model.
        
        Args:
            vocab_size: int The size of the vocabulary
            context_length: int The maximum context length  
            d_model: int Hidden dimension of the model
            num_layers: int The number of Transformer blocks
            num_heads: int Number of attention heads
            d_ff: int Dimension of the feed-forward layer
            rope_theta: float RoPE parameter
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.device = device
        self.dtype = dtype
        
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer language model.
        
        Args:
            input_ids: torch.Tensor of shape (batch_size, sequence_length)
        
        Returns:
            torch.Tensor of shape (batch_size, sequence_length, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Generate token positions for RoPE
        token_positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Token embeddings
        x = self.token_embeddings(input_ids)
        
        for layer in self.layers:
            x = layer(x, token_positions)
        
        x = self.ln_final(x)
        
        logits = self.lm_head(x)
        
        return logits

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute the cross-entropy loss between inputs and targets in a numerically stable way.
    
    Args:
        inputs: torch.Tensor of shape (batch_size, vocab_size) with unnormalized logits
        targets: torch.Tensor of shape (batch_size,) with target class indices
        
    Returns:
        torch.Tensor: scalar tensor with the average cross-entropy loss
    """
    batch_size, vocab_size = inputs.shape
    
    # Compute log_softmax in a numerically stable way
    # log_softmax(x) = x - log(sum(exp(x)))
    #                = x - x_max - log(sum(exp(x - x_max)))
    x_max = torch.max(inputs, dim=-1, keepdim=True).values
    x_shifted = inputs - x_max
    log_sum_exp = torch.log(torch.sum(torch.exp(x_shifted), dim=-1, keepdim=True))
    log_softmax = x_shifted - log_sum_exp
    
    # Select the log probabilities for the target classes
    # log_softmax has shape (batch_size, vocab_size)
    # targets has shape (batch_size,)
    # We want to select log_softmax[i, targets[i]] for each i
    target_log_probs = log_softmax[torch.arange(batch_size), targets]
    
    # Cross-entropy loss is the negative log probability, averaged over the batch
    cross_entropy_loss = -torch.mean(target_log_probs)
    
    return cross_entropy_loss

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Cosine learning rate schedule with linear warmup.
    
    Args:
        it: Current iteration number
        max_learning_rate: Maximum learning rate (alpha_max)
        min_learning_rate: Minimum learning rate (alpha_min)  
        warmup_iters: Number of warmup iterations (T_w)
        cosine_cycle_iters: Total iterations for the entire schedule (T_c)
        
    Returns:
        Learning rate at the given iteration
    """
    import math
    
    if it <= warmup_iters:
        # Linear warmup phase: linearly increase from 0 to max_learning_rate
        return max_learning_rate * it / warmup_iters
    elif it <= cosine_cycle_iters:
        # Cosine annealing phase: cosine decay from max_learning_rate to min_learning_rate
        # The actual cosine decay length is (cosine_cycle_iters - warmup_iters)
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + (max_learning_rate - min_learning_rate) * 0.5 * (1 + math.cos(math.pi * progress))
    else:
        # Post-annealing phase: maintain min_learning_rate
        return min_learning_rate

def gradient_clipping(parameters, max_l2_norm: float) -> None:
    """
    Clip gradients to have L2 norm at most max_l2_norm.
    
    Args:
        parameters: Iterable of parameters with gradients to clip
        max_l2_norm: Maximum L2 norm for the gradients
        
    The gradients are modified in-place.
    """
    import math
    
    eps = 1e-6
    
    # Collect all parameters that have gradients
    grads = []
    for p in parameters:
        if p.grad is not None:
            grads.append(p.grad)
    
    if not grads:
        return
    
    total_norm = 0.0
    for grad in grads:
        param_norm = grad.data.norm(dtype=torch.float32)
        total_norm += param_norm.item() ** 2
    total_norm = math.sqrt(total_norm)
    
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)
        for grad in grads:
            grad.data.mul_(clip_coef)


def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
                           Can be a regular numpy array or memory-mapped array (np.memmap).
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # Validate that we have enough data for the requested context_length
    if len(dataset) <= context_length:
        raise ValueError(f"Dataset size ({len(dataset)}) must be larger than context_length ({context_length}). "
                        f"Either use a larger dataset or reduce context_length to less than {len(dataset)}.")
    
    max_start_idx = len(dataset) - context_length
    
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)
    
    input_sequences = np.zeros((batch_size, context_length), dtype=np.int64)
    target_sequences = np.zeros((batch_size, context_length), dtype=np.int64)
    
    for i in range(batch_size):
        start_idx = start_indices[i]
        input_sequences[i] = dataset[start_idx:start_idx + context_length]
        target_sequences[i] = dataset[start_idx + 1:start_idx + context_length + 1]
    
    input_tensor = torch.from_numpy(input_sequences).to(device)
    target_tensor = torch.from_numpy(target_sequences).to(device)
    
    return input_tensor, target_tensor


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out) -> None:
    """
    Save model state, optimizer state, and iteration number to a checkpoint file.
    
    Args:
        model: torch.nn.Module to save
        optimizer: torch.optim.Optimizer to save  
        iteration: int iteration number to save
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes] output file path or file-like object
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)


def load_checkpoint(src, model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """
    Load checkpoint from file and restore model and optimizer states.
    
    Args:
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes] source file path or file-like object
        model: torch.nn.Module to restore state to
        optimizer: torch.optim.Optimizer to restore state to
        
    Returns:
        int: iteration number from the checkpoint
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']


def decode(
    model: TransformerLM,
    prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    endoftext_token_id: int = None,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Generate text from a language model using temperature scaling and top-p sampling.
    
    Args:
        model: The TransformerLM model to use for generation
        prompt: torch.Tensor of shape (batch_size, prompt_length) containing input token IDs
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Temperature value for softmax scaling (higher = more random)
        top_p: Cumulative probability threshold for nucleus sampling (1.0 = no filtering)
        endoftext_token_id: Token ID for end-of-text (if None, generate max_new_tokens)
        device: Device to run generation on
        
    Returns:
        torch.Tensor of shape (batch_size, prompt_length + generated_length) with generated sequences
    """
    model.eval()
    
    if prompt.device != device:
        prompt = prompt.to(device)
    
    generated = prompt.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated)
            next_token_logits = logits[:, -1, :]
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                for batch_idx in range(probs.shape[0]):
                    indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                    probs[batch_idx, indices_to_remove] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if endoftext_token_id is not None:
                # next_token shape: [batch_size, 1], so we check if any batch contains endoftext
                if (next_token.squeeze(-1) == endoftext_token_id).any():
                    break
    
    model.train()
    return generated


def temperature_sample(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits and sample from the resulting distribution.
    
    # Temperature scaling is a technique to control the randomness of generation:
    # - temperature < 1.0: Makes the distribution sharper, favoring high-probability tokens (more deterministic)
    # - temperature = 1.0: Keeps the original distribution
    # - temperature > 1.0: Makes the distribution flatter, increasing the chance of selecting low-probability tokens (more random)
    
    Args:
        logits: torch.Tensor of shape (..., vocab_size) with unnormalized log probabilities
        temperature: Temperature value for scaling
        
    Returns:
        torch.Tensor of shape (...,) with sampled token indices
    """
    if temperature == 0:
        return torch.argmax(logits, dim=-1)
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1).view(logits.shape[:-1])


def top_p_sample(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply top-p (nucleus) sampling to logits.
    
    # The core idea of top-p sampling (also known as nucleus sampling) is:
    # 1. Sort the vocabulary by probability in descending order
    # 2. Compute the cumulative probability until it exceeds the threshold p
    # 3. Only sample from this "nucleus" subset of tokens, ignoring the remaining low-probability tokens
    #
    # This method dynamically adjusts the candidate set size, preserving diversity while avoiding sampling extremely low-probability tokens.
    
    Args:
        logits: torch.Tensor of shape (..., vocab_size) with unnormalized log probabilities
        p: Cumulative probability threshold (typically 0.9-0.95)
        temperature: Temperature value for scaling
        
    Returns:
        torch.Tensor of shape (...,) with sampled token indices
    """
    if temperature != 1.0:
        logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    keep_mask = cumulative_probs <= p
    keep_mask[..., 1:] = keep_mask[..., :-1].clone()
    keep_mask[..., 0] = True
    filtered_probs = sorted_probs.clone()
    filtered_probs[~keep_mask] = 0
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    sample_indices = torch.multinomial(filtered_probs.view(-1, filtered_probs.shape[-1]), num_samples=1)
    original_indices = torch.gather(
        sorted_indices.view(-1, sorted_indices.shape[-1]), 
        1, 
        sample_indices
    )
    return original_indices.view(logits.shape[:-1])


