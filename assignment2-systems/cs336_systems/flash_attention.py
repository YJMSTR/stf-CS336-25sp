import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce


class FlashAttention2PyTorch(torch.autograd.Function):
    """
    FlashAttention-2 implementation using pure PyTorch operations.
    
    This is a slower but more debuggable implementation compared to the Triton version.
    It implements the tiled attention algorithm with proper memory management.
    """
    
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch_size, n_queries, d = Q.shape
        n_keys = K.shape[-2]
        B_q = 16
        B_k = 16
        out = torch.zeros_like(Q, device=Q.device, dtype=Q.dtype)
        logsumexp = torch.zeros(batch_size, n_queries, device=Q.device, dtype=Q.dtype)
        for i in range(0, n_queries, B_q):
            Q_tile = Q[:, i:i+B_q, :]
            out_i = torch.zeros_like(Q_tile, device=Q.device, dtype=Q.dtype)
            l_i = torch.zeros(batch_size, B_q, device=Q.device, dtype=Q.dtype)
            m_i = torch.full((batch_size, B_q), float('-inf'), device=Q.device, dtype=Q.dtype)
            for j in range(0, n_keys, B_k):
                K_tile = K[:, j:j+B_k, :]
                V_tile = V[:, j:j+B_k, :]
                S_ij = Q_tile @ K_tile.transpose(-2, -1)
                S_ij = S_ij / (d ** 0.5)
                m_ij = torch.max(m_i, reduce(S_ij, 'b q k -> b q', 'max')) # (batch_size, B_q)
                P_ij = torch.exp(S_ij - rearrange(m_ij, 'b q -> b q 1'))
                l_ij = torch.exp(m_i - m_ij) * l_i + reduce(P_ij, 'b q k -> b q', 'sum') # (batch_size, B_q)
                out_ij = out_i * rearrange(torch.exp(m_i - m_ij), 'b q -> b q 1') + P_ij @ V_tile # (batch_size, B_q, B_q) @ (batch_size, B_q, d) 
                out_i = out_ij
                l_i = l_ij
                m_i = m_ij
            out_i = out_i * rearrange(1.0 / l_i, 'b q -> b q 1')
            L_i = m_i + torch.log(l_i)
            out[:, i:i+B_q, :] = out_i
            logsumexp[:, i:i+B_q] = L_i

        ctx.save_for_backward(logsumexp, Q, K, V, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


def get_flashattention_autograd_function_pytorch():
    return FlashAttention2PyTorch
