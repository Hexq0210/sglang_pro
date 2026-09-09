"""Check per-request MLA prefix expansion against independent dense attention.

CPU tests replace FIA with a dense oracle. Set SGLANG_TEST_DEVICE=npu:8 to
exercise real FIA with the same paged-prefix, padding and call-count checks.
"""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ModuleNotFoundError:

    def register_cpu_ci(*args, **kwargs):
        pass


register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/hardware_backend/npu/attention/mla_cache.py"
)
_SPEC = importlib.util.spec_from_file_location("mla_cache_under_test", _SOURCE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def dense_attention(q, k, v, scale, causal):
    scores = q.float().transpose(1, 2) @ k.float().transpose(1, 2).transpose(-1, -2)
    scores *= scale
    if causal:
        mask = torch.ones(
            q.shape[1], k.shape[1], dtype=torch.bool, device=q.device
        ).triu(k.shape[1] - q.shape[1] + 1)
        scores.masked_fill_(mask, -torch.inf)
    return (scores.softmax(-1) @ v.float().transpose(1, 2)).transpose(1, 2)


class TestNDMLAPrefixAttention(unittest.TestCase):
    is_nz = False

    @classmethod
    def setUpClass(cls):
        cls.device = os.environ.get("SGLANG_TEST_DEVICE", "cpu")
        cls.dtype = torch.float32
        if cls.device.startswith("npu"):
            import torch_npu  # noqa: F401

            torch.npu.set_device(cls.device)
            cls.dtype = torch.bfloat16

    def check_case(self, prefix_lens, extend_lens, *, page_size=128, logit_scale=1.0):
        torch.manual_seed(7)
        heads, latent_dim, nope_dim, rope_dim, v_dim = 12, 512, 128, 64, 128
        kwargs = {"device": self.device, "dtype": self.dtype}
        max_prefix = max(prefix_lens, default=0)
        num_pages = (max_prefix + page_size - 1) // page_size
        latent = torch.randn(num_pages, page_size, 1, latent_dim, **kwargs)
        rope = torch.randn(num_pages, page_size, 1, rope_dim, **kwargs)
        # Reuse and reorder physical pages, including a sentinel-filled tail.
        pages = torch.arange(num_pages - 1, -1, -1, device=self.device)
        tables = torch.cat(
            [pages[: (p + page_size - 1) // page_size] for p in prefix_lens]
        ).int()
        if max_prefix and max_prefix % page_size:
            physical_page = num_pages - 1 - max_prefix // page_size
            latent[physical_page, max_prefix % page_size :] = 100
            rope[physical_page, max_prefix % page_size :] = 100

        def pack(cache):
            if not self.is_nz:
                return cache
            return (
                cache.reshape(num_pages, page_size, cache.shape[-1] // 16, 16)
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape_as(cache)
            )

        weight = (
            torch.randn(heads * (nope_dim + v_dim), latent_dim, **kwargs)
            / latent_dim**0.5
        )
        projection_sizes = []

        def project(x):
            projection_sizes.append(x.numel() // latent_dim)
            return torch.nn.functional.linear(x, weight), None

        ntokens = sum(extend_lens)
        q = torch.randn(ntokens + 3, heads, nope_dim + rope_dim, **kwargs)
        current_kv = torch.randn(ntokens + 3, heads, nope_dim + v_dim, **kwargs)
        current_rope = torch.randn(ntokens + 3, heads, rope_dim, **kwargs)
        k = torch.cat([current_kv[..., :nope_dim], current_rope], dim=-1)
        v = current_kv[..., nope_dim:]  # Match the production noncontiguous V view.
        mask = torch.ones(2048, 2048, dtype=torch.bool, device=self.device).triu(1)
        scale = logit_scale / (nope_dim + rope_dim) ** 0.5
        fia_calls = []
        real_fia = (
            torch.ops.npu.npu_fused_infer_attention_score
            if self.device != "cpu"
            else None
        )

        def run_fia(q_req, k_req, v_req, **options):
            fia_calls.append(q_req.shape[1])
            self.assertEqual(options["input_layout"], "BSND")
            self.assertFalse(options.get("softmax_lse_flag", False))
            if real_fia is not None:
                return real_fia(q_req, k_req, v_req, **options)
            out = dense_attention(
                q_req,
                k_req,
                v_req,
                options["scale"],
                options.get("sparse_mode", 0) == 3,
            )
            return out.to(q_req.dtype), None

        with patch.object(
            torch.ops.npu, "npu_fused_infer_attention_score", run_fia, create=True
        ):
            actual = _MODULE.per_request_mla_prefix_attention(
                q,
                k,
                v,
                k_buffer=pack(latent),
                v_buffer=pack(rope),
                prefix_block_tables=tables,
                prefix_lens=prefix_lens,
                extend_lens=extend_lens,
                page_size=page_size,
                kv_b_proj=project,
                qk_nope_head_dim=nope_dim,
                scale=scale,
                causal_mask=mask,
                is_nz=self.is_nz,
            )
        # Exactly one FIA per nonempty request, and no batch-wide projection.
        self.assertEqual(fia_calls, [s for s in extend_lens if s])
        self.assertEqual(
            projection_sizes, [p for p, s in zip(prefix_lens, extend_lens) if p and s]
        )

        offset = 0
        for prefix_len, qlen in zip(prefix_lens, extend_lens):
            if qlen == 0:
                continue
            ids = pages[: (prefix_len + page_size - 1) // page_size].long()
            cache = latent.index_select(0, ids).flatten(0, 1)[:prefix_len]
            cache_rope = rope.index_select(0, ids).flatten(0, 1)[:prefix_len]
            expanded = torch.nn.functional.linear(cache, weight).view(
                prefix_len, heads, nope_dim + v_dim
            )
            prefix_k = torch.cat(
                [expanded[..., :nope_dim], cache_rope.expand(-1, heads, -1)], dim=-1
            )
            full_k = torch.cat([prefix_k, k[offset : offset + qlen]], dim=0)
            full_v = torch.cat(
                [expanded[..., nope_dim:], v[offset : offset + qlen]], dim=0
            )
            expected = dense_attention(
                q[None, offset : offset + qlen], full_k[None], full_v[None], scale, True
            )
            tolerance = 0.015 if self.dtype == torch.bfloat16 else 1e-5
            torch.testing.assert_close(
                actual[offset : offset + qlen].float(),
                expected.squeeze(0),
                atol=tolerance,
                rtol=tolerance,
            )
            offset += qlen
        self.assertEqual(actual[ntokens:].count_nonzero().item(), 0)

    def test_mixed_hits_shared_prefix_partial_pages_and_empty_request(self):
        self.check_case([257, 0, 513, 257, 129], [7, 5, 0, 1, 3])

    def test_aligned_prefixes(self):
        self.check_case([1024, 512], [9, 7])

    def test_all_misses(self):
        self.check_case([0, 0], [7, 1])

    def test_large_logits(self):
        self.check_case([513], [3], logit_scale=8.0)

    def test_small_pages(self):
        self.check_case([385, 256], [7, 1], page_size=16)

    def test_query_longer_than_prefix(self):
        self.check_case([129], [257])


class TestNZMLAPrefixAttention(TestNDMLAPrefixAttention):
    is_nz = True


if __name__ == "__main__":
    unittest.main()
