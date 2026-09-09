"""Read logical pages from the explicit PA-NZ storage of the NPU MLA cache."""

from typing import Callable, Sequence

import torch


def gather_mla_cache_pages(
    cache: torch.Tensor, block_ids: torch.Tensor, *, is_nz: bool
) -> torch.Tensor:
    """Return selected pages in logical [blocks, page_size, 1, head_dim] order.

    NZ buffers retain that public shape, but their physical contents are
    [blocks, head_dim // 16, page_size, 16]. Restore token-major order before
    projecting cached latent vectors or concatenating their RoPE features.
    """
    pages = torch.index_select(cache, 0, block_ids)
    if not is_nz:
        return pages
    page_size, head_dim = cache.shape[1], cache.shape[-1]
    return (
        pages.view(block_ids.numel(), head_dim // 16, page_size, 16)
        .permute(0, 2, 1, 3)
        .reshape(block_ids.numel(), page_size, 1, head_dim)
    )


# Keep the existing batch projection for small prefixes to avoid extra GEMM launches.
MLA_PREFIX_BATCH_EXPAND_LIMIT = 4096


def _request_prefix_attention(
    q,
    k,
    v,
    k_buffer,
    v_buffer,
    blocks,
    prefix_len,
    kv_b_proj,
    qk_nope_head_dim,
    scale,
    causal_mask,
    is_nz,
):
    # Expanded KV lives only until this request's FIA finishes being enqueued.
    # Returning only the output prevents the next request from retaining it.
    if prefix_len:
        latent = gather_mla_cache_pages(k_buffer, blocks, is_nz=is_nz).flatten(0, 1)[
            :prefix_len
        ]
        rope = gather_mla_cache_pages(v_buffer, blocks, is_nz=is_nz).flatten(0, 1)[
            :prefix_len
        ]
        kv = kv_b_proj(latent)[0].view(
            prefix_len, k.shape[1], qk_nope_head_dim + v.shape[-1]
        )
        k_nope, v_prefix = kv.split([qk_nope_head_dim, v.shape[-1]], dim=-1)
        k_prefix = torch.cat([k_nope, rope.expand(-1, k.shape[1], -1)], dim=-1)
        k = torch.cat([k_prefix, k], dim=0)
        v = torch.cat([v_prefix, v], dim=0)
    masked = prefix_len > 0 or q.shape[0] > 1
    return torch.ops.npu.npu_fused_infer_attention_score(
        q.unsqueeze(0).contiguous(),
        k.unsqueeze(0).contiguous(),
        v.unsqueeze(0).contiguous(),
        num_heads=q.shape[1],
        num_key_value_heads=k.shape[1],
        input_layout="BSND",
        atten_mask=causal_mask if masked else None,
        sparse_mode=3 if masked else 0,
        scale=scale,
        next_tokens=0,
    )[0].squeeze(0)


def per_request_mla_prefix_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    prefix_block_tables: torch.Tensor,
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    page_size: int,
    kv_b_proj: Callable,
    qk_nope_head_dim: int,
    scale: float,
    causal_mask: torch.Tensor,
    is_nz: bool,
) -> torch.Tensor:
    """Expand one request's prefix at a time, with one FIA call per request.

    This removes the batch-size multiplier from expanded-prefix memory without
    introducing attention splits or LSE merging. Memory still scales with the
    longest individual prefix. Cache buffers are [pages, page_size, 1, dim].
    """
    output = q.new_zeros((q.shape[0], q.shape[1], v.shape[-1]))
    q_offset = block_offset = 0
    for prefix_len, q_len in zip(prefix_lens, extend_lens):
        num_blocks = (prefix_len + page_size - 1) // page_size
        blocks = prefix_block_tables[block_offset : block_offset + num_blocks]
        block_offset += num_blocks
        if q_len == 0:
            continue
        end = q_offset + q_len
        output[q_offset:end] = _request_prefix_attention(
            q[q_offset:end],
            k[q_offset:end],
            v[q_offset:end],
            k_buffer,
            v_buffer,
            blocks,
            prefix_len,
            kv_b_proj,
            qk_nope_head_dim,
            scale,
            causal_mask,
            is_nz,
        )
        q_offset = end
    return output
