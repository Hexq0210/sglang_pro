"""Complete, request-aligned proposals carried to the next DSPARK verify."""

from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2


def deterministic_draft_logits(tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Logits for the point-mass proposal used to bootstrap the pipeline.

    Rejection sampling must use the distribution that actually produced the
    mock tokens. Repeating the anchor is a deterministic proposal, not a draw
    from the Markov head (nor from a uniform distribution).
    """
    logits = torch.full(
        (*tokens.shape, vocab_size),
        -torch.inf,
        dtype=torch.float32,
        device=tokens.device,
    )
    return logits.scatter_(-1, tokens.unsqueeze(-1).long(), 0.0)


@dataclass
class DSparkDraftInputV2(DFlashDraftInputV2):
    prefetched_tokens: Optional[torch.Tensor] = None
    prefetched_logits: Optional[torch.Tensor] = None
    prefetched_confidence: Optional[torch.Tensor] = None
    prefetched_confidence_raw: Optional[torch.Tensor] = None
    prefetched_confidence_valid: Optional[torch.Tensor] = None

    def init_mock_proposal(self, gamma: int) -> None:
        # The root remains the real bonus token. These repeated candidates are
        # verified normally; no draft forward is needed on the first decode.
        self.prefetched_tokens = self.bonus_tokens.view(-1, 1).repeat(1, gamma)
        self.prefetched_logits = None  # materialize the point mass only if needed
        self.prefetched_confidence = torch.zeros(
            self.prefetched_tokens.shape,
            dtype=torch.float32,
            device=self.bonus_tokens.device,
        )

        self.prefetched_confidence_raw = None
        self.prefetched_confidence_valid = torch.zeros(
            self.prefetched_tokens.shape[0],
            dtype=torch.bool,
            device=self.bonus_tokens.device,
        )

    def store_prefetched(
        self, tokens, logits, confidence, *, confidence_raw=None
    ) -> None:
        # Own graph outputs before a subsequent replay reuses their storage.
        self.prefetched_tokens = tokens.clone()
        self.prefetched_logits = None if logits is None else logits.clone()
        self.prefetched_confidence = (
            torch.zeros_like(tokens, dtype=torch.float32)
            if confidence is None
            else confidence.clone()
        )
        self.prefetched_confidence_raw = (
            None if confidence_raw is None else confidence_raw.clone()
        )
        self.prefetched_confidence_valid = torch.full(
            (tokens.shape[0],),
            confidence_raw is not None,
            dtype=torch.bool,
            device=tokens.device,
        )

    def _record_prefetch_stream(self) -> None:
        for tensor in (
            self.prefetched_tokens,
            self.prefetched_logits,
            self.prefetched_confidence,
            self.prefetched_confidence_raw,
            self.prefetched_confidence_valid,
        ):
            if tensor is not None and tensor.device.type != "cpu":
                tensor.record_stream(
                    torch.get_device_module(tensor.device).current_stream()
                )

    def take_prefetched(self):
        self._record_prefetch_stream()
        tensors = (
            self.prefetched_tokens,
            self.prefetched_logits,
            self.prefetched_confidence,
            self.prefetched_confidence_raw,
            self.prefetched_confidence_valid,
        )
        self.prefetched_tokens = None
        self.prefetched_logits = None
        self.prefetched_confidence = None
        self.prefetched_confidence_raw = None
        self.prefetched_confidence_valid = None
        return tensors

    def filter_batch(
        self,
        new_indices: torch.Tensor,
        new_indices_cpu: Optional[List[int]] = None,
    ):
        # FutureMap's early return filters only future_indices in the base.
        self._record_prefetch_stream()
        for name in (
            "prefetched_tokens",
            "prefetched_logits",
            "prefetched_confidence",
            "prefetched_confidence_raw",
            "prefetched_confidence_valid",
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value[new_indices])
        super().filter_batch(new_indices, new_indices_cpu)

    def merge_batch(self, spec_info: DFlashDraftInputV2):
        self._record_prefetch_stream()
        if isinstance(spec_info, DSparkDraftInputV2):
            spec_info._record_prefetch_stream()
        other_tokens = getattr(spec_info, "prefetched_tokens", None)
        if self.prefetched_tokens is not None or other_tokens is not None:
            assert self.prefetched_tokens is not None and other_tokens is not None
            left, right = self.prefetched_logits, spec_info.prefetched_logits
            if left is not None or right is not None:
                # A cold arrival has a point-mass q; an all-greedy batch has no
                # q because its rows use greedy accept even after a mixed merge.
                vocab_size = (left if left is not None else right).shape[-1]
                if left is None:
                    left = deterministic_draft_logits(
                        self.prefetched_tokens, vocab_size
                    )
                if right is None:
                    right = deterministic_draft_logits(other_tokens, vocab_size)
                self.prefetched_logits = torch.cat([left, right], dim=0)
            left_raw, right_raw = (
                self.prefetched_confidence_raw,
                spec_info.prefetched_confidence_raw,
            )
            if left_raw is not None or right_raw is not None:
                raw_dtype = (left_raw if left_raw is not None else right_raw).dtype
                if left_raw is None:
                    left_raw = torch.zeros_like(self.prefetched_tokens, dtype=raw_dtype)
                if right_raw is None:
                    right_raw = torch.zeros_like(other_tokens, dtype=raw_dtype)
                self.prefetched_confidence_raw = torch.cat([left_raw, right_raw], dim=0)
            self.prefetched_confidence_valid = torch.cat(
                [
                    self.prefetched_confidence_valid,
                    spec_info.prefetched_confidence_valid,
                ]
            )
            self.prefetched_tokens = torch.cat(
                [self.prefetched_tokens, other_tokens], dim=0
            )
            self.prefetched_confidence = torch.cat(
                [self.prefetched_confidence, spec_info.prefetched_confidence], dim=0
            )
        super().merge_batch(spec_info)
