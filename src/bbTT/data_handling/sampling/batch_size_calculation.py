from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from bbTT.data_handling.sampling.sample_strategy import RoundingStrategy


class BatchSizeAllocatorOld:
    """
    Encapsulates the largest-remainder-style rounding algorithm used to split a
    sub-batch size across process ids proportional to their normalization weight,
    respecting a minimum per-process size. Pure function of plain data in,
    (sizes, relative_weights) out — testable without constructing Process objects.
    """

    def __init__(self, min_size: int = 0):
        self.min_size = min_size

    def allocate(
        self,
        weights_by_pid: dict[str, float],
        sub_sample_ratio: dict[str, float],
        sub_batch_size: int,
        sample_ratio_for_type: float,
    ) -> tuple[dict[str, int], dict[str, float]]:
        """
        Returns:
            sizes: process_id -> integer event count, summing to sub_batch_size.
            relative_weights: process_id -> relative contribution to the batch.
        """
        pids = list(weights_by_pid.keys())
        raw_weights = torch.tensor(
            [weights_by_pid[pid] * sub_sample_ratio.get(pid, 1) for pid in pids], dtype=torch.float32
        )
        total_weight = raw_weights.sum()

        relative_weights = {pid: (w / total_weight * sample_ratio_for_type).item() for pid, w in zip(pids, raw_weights)}

        ideal = sub_batch_size * raw_weights / total_weight
        floored = torch.maximum(torch.floor(ideal), torch.tensor(float(self.min_size)))
        overflow = int(floored.sum()) - sub_batch_size

        above_mask = floored > self.min_size
        above = floored[above_mask]
        overflow_shares = above / above.sum() * overflow
        overflow_shares, order = torch.sort(overflow_shares)

        median = order.median()

        below_median = order <= median if len(order) % 2 == 0 else order < median
        above_median = ~below_median

        overflow_shares[below_median] = torch.floor(overflow_shares[below_median])
        overflow_shares[above_median] = torch.ceil(overflow_shares[above_median])

        above_indices = torch.arange(len(floored))[above_mask][order]
        floored[above_indices] -= overflow_shares

        if (floored.to(torch.int32).sum() - floored.sum()) != 0:
            raise TypeError("Rounding changed the total value unexpectedly")
        floored = floored.to(torch.int32)

        remainder = int(floored.sum()) - sub_batch_size
        floored[above_indices[0]] -= remainder

        sizes = {pid: n.item() for pid, n in zip(pids, floored)}
        return sizes, relative_weights


class BatchSizeAllocator:
    """
    Splits a sub-batch size across process ids proportional to their
    normalization weight. The rounding behaviour is pluggable via
    RoundingStrategy.
    """

    def __init__(self, rounding: RoundingStrategy):
        self.rounding = rounding

    def allocate(
        self,
        weights_by_pid: dict[str, float],
        sub_sample_ratio: dict[str, float],
        sub_batch_size: int,
        sample_ratio_for_type: float,
    ) -> tuple[dict[str, int], dict[str, float]]:
        """
        Args:
            weights_by_pid (dict[str, float]): Cross-section weight per process id.
            sub_sample_ratio (dict[str, float]): Extra per-pid scale factor,
                defaults to 1 if a pid is absent.
            sub_batch_size (int): Events to allocate across all pids.
            sample_ratio_for_type (float): Parent-level share this sub-batch
                belongs to, folded into relative_weights.
            generator (torch.Generator, optional): For reproducible draws;
                ignored by deterministic strategies.

        Returns:
            sizes: process_id -> integer event count, summing exactly to sub_batch_size.
            relative_weights: process_id -> relative contribution to the batch.
        """
        pids = list(weights_by_pid.keys())

        # for oversampling certain processes raw weights are scaled
        raw_weights = torch.tensor(
            [weights_by_pid[pid] * sub_sample_ratio.get(pid, 1) for pid in pids], dtype=torch.float32
        )
        total_weight = raw_weights.sum()

        # expected relative contribution of a pid to process BEFORE apply rounding strategy.
        # this is necessary for validation losses. A batch is inflated by oversampling.
        # but validation does not use oversampling inflation, but real one.
        # this is only driven by cross section and sample ratio
        relative_weights = {pid: (w / total_weight * sample_ratio_for_type).item() for pid, w in zip(pids, raw_weights)}

        # normalize by total weights ensures consistent process family number
        exact = sub_batch_size * raw_weights / total_weight
        counts = self.rounding.round(exact)

        sizes = {pid: n.item() for pid, n in zip(pids, counts)}

        # relative contribution of a pid for the batch AFTER apply rounding strategy
        # depending on the rounding strategy relative_weight does not represent real presence
        # e.g. flooring (and no stochastic strategy is used), then relative_weight is calculated with wrong quantity
        post_relative_weights = {
            pid: (num_per_pid / sub_batch_size * sample_ratio_for_type) for pid, num_per_pid in sizes.items()
        }
        return sizes, relative_weights, post_relative_weights
