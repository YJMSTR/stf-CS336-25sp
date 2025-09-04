import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
import triton
import triton.language as tl



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

def get_flashattention_autograd_function_triton():
    return FlashAttention2Triton


def maybe_contiguous(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_contiguous() else x.contiguous()

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vv, stride_vd,
    stride_ob, stride_oo, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vv, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oo, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    m_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32) - float("inf")
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    acc_o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    query_tile = tl.load(Q_block_ptr, boundary_check=(0, 1))
    # It's good practice to apply the scale factor (1/sqrt(d_k)) to q right after loading it.
    query_tile = (query_tile * scale).to(tl.float32)

    for j in range(0, N_KEYS, K_TILE_SIZE):
        key_tile = tl.load(K_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        value_tile = tl.load(V_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        S_ij = tl.dot(query_tile, key_tile.T)
        if is_causal:
            row_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            col_idx = j  + tl.arange(0, K_TILE_SIZE)
            mask = col_idx[None, :] > row_idx[:, None]
            S_ij = tl.where(mask, -float('inf'), S_ij)
        m_ij = tl.maximum(m_i, tl.max(S_ij, axis=1))
        P_ij = tl.exp(S_ij - m_ij[:, None])
        l_ij = tl.exp(m_i - m_ij) * l_i + tl.sum(P_ij, axis=1)
        acc_o_ij = acc_o * tl.exp(m_i - m_ij)[:, None] + tl.dot(P_ij, value_tile)
        acc_o = acc_o_ij
        l_i = l_ij
        m_i = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    acc_o = acc_o * (1.0 / l_i)[:, None]
    logexpsum = m_i + tl.log(l_i)
    tl.store(O_block_ptr, acc_o.to(O_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, logexpsum, boundary_check=(0))



class FlashAttention2Triton(torch.autograd.Function):
    """
    Your launch grid should be set as (Tq, batch_size), meaning each Triton program instance
    will load only elements from a single batch index, and only read/write to a single query tile
    of Q, O, and L.
    """
    def forward(ctx, Q, K, V, is_causal=False):
        batch_size, n_queries, d = Q.shape
        n_keys = K.shape[-2]
        B_q = 16
        B_k = 16
        out = torch.zeros_like(Q, device=Q.device, dtype=Q.dtype)
        # use float32 for better precision
        logsumexp = torch.zeros(batch_size, n_queries, device=Q.device, dtype=torch.float32)
        Q = maybe_contiguous(Q)
        K = maybe_contiguous(K)
        V = maybe_contiguous(V)
        stride_qb, stride_qq, stride_qd = Q.stride()
        stride_kb, stride_kk, stride_kd = K.stride()
        stride_vb, stride_vv, stride_vd = V.stride()
        stride_ob, stride_oo, stride_od = out.stride()
        stride_lb, stride_lq = logsumexp.stride()
        scale = 1 / (d ** 0.5)
        grid = (triton.cdiv(n_queries, B_q), batch_size)
        flash_fwd_kernel[grid](
            Q, K, V, out, logsumexp,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vv, stride_vd,
            stride_ob, stride_oo, stride_od,
            stride_lb, stride_lq,
            n_queries, n_keys,
            scale,
            d,
            B_q, B_k,
            is_causal
        )
        ctx.save_for_backward(logsumexp, Q, K, V, out)
        ctx.is_causal = is_causal
        return out

    def backward(ctx, grad_output):
        pass