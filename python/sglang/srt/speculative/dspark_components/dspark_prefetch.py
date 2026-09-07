"""Request-aligned backbone outputs carried to the next DSPARK iteration."""

from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2


@dataclass
class DSparkDraftInputV2(DFlashDraftInputV2):
    # Both tensors have a leading request dimension. No logits, sampled tokens,
    # RNG state or verify budget is carried ahead of its normal iteration.
    prefetched_hidden: Optional[torch.Tensor] = None
    prefetched_block_ids: Optional[torch.Tensor] = None

    def _record_prefetch_stream(self) -> None:
        for tensor in (self.prefetched_hidden, self.prefetched_block_ids):
            if tensor is not None and tensor.device.type != "cpu":
                tensor.record_stream(
                    torch.get_device_module(tensor.device).current_stream()
                )

    def take_prefetched(self):
        self._record_prefetch_stream()
        hidden, block_ids = self.prefetched_hidden, self.prefetched_block_ids
        self.prefetched_hidden = self.prefetched_block_ids = None
        return hidden, block_ids

    def filter_batch(
        self,
        new_indices: torch.Tensor,
        new_indices_cpu: Optional[List[int]] = None,
    ):
        # FutureMap's early-return path filters only future_indices in the base
        # class. These local tensors must follow the same request permutation.
        self._record_prefetch_stream()
        if self.prefetched_hidden is not None:
            self.prefetched_hidden = self.prefetched_hidden[new_indices]
            self.prefetched_block_ids = self.prefetched_block_ids[new_indices]
        super().filter_batch(new_indices, new_indices_cpu)

    def merge_batch(self, spec_info: DFlashDraftInputV2):
        self._record_prefetch_stream()
        if isinstance(spec_info, DSparkDraftInputV2):
            spec_info._record_prefetch_stream()
        other_hidden = getattr(spec_info, "prefetched_hidden", None)
        if self.prefetched_hidden is not None and other_hidden is not None:
            self.prefetched_hidden = torch.cat(
                [self.prefetched_hidden, other_hidden], dim=0
            )
            self.prefetched_block_ids = torch.cat(
                [self.prefetched_block_ids, spec_info.prefetched_block_ids], dim=0
            )
        else:
            # A prefill/PD arrival has no cached backbone output. Recompute the
            # complete merged batch; never mix cached rows with uninitialized ones.
            self.prefetched_hidden = self.prefetched_block_ids = None
        super().merge_batch(spec_info)
