from abc import ABC, abstractmethod

import torch


class RoundingStrategy(ABC):
    """
    Turns exact (non-integer) per-pid shares of a batch into integer counts
    summing exactly to the batch size.

    Kept separate from BatchSizeAllocator so the two rounding behaviors can be
    swapped without touching the weight computation they share.
    """

    @abstractmethod
    def round(
        self,
        exact: torch.Tensor,
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        """
        Args:
            exact (torch.Tensor): Ideal (possibly fractional) count per pid.
            generator (torch.Generator, optional): For reproducible draws;
                unused by deterministic strategies.

        Returns:
            torch.Tensor: Integer counts, same length as exact, summing
            exactly to round(exact.sum()).
        """

    @abstractmethod
    def expected(self, exact: torch.Tensor) -> torch.Tensor:
        """
        Expected count per pid when ``round`` is called repeatedly on *exact*.

        Kept next to ``round`` because only the strategy knows whether its rounding is
        unbiased. The expected count is what a loss evaluated on unsampled data has to
        reproduce, so it must not depend on a single draw.

        Args:
            exact (torch.Tensor): Ideal (possibly fractional) count per pid.

        Returns:
            torch.Tensor: Expected count per pid, same length and dtype as exact.
        """


class LargestRemainderRounding(RoundingStrategy):
    """
    Floors every share, then guarantees a minimum per pid and claws the
    resulting overflow back from the least-clamped pids.

    Deterministic and reproducible, but the minimum systematically
    overrepresented any pid whose ideal share is below min_size — a pid at
    0.2 always gets min_size, not 0.2 on average.
    """

    def __init__(self, min_size: int = 0):
        self.min_size = min_size

    def expected(self, exact: torch.Tensor) -> torch.Tensor:
        # deterministic: every call returns the same counts, so the expectation is the rounding itself
        return self.round(exact).to(exact.dtype)

    def round(
        self,
        exact: torch.Tensor,
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        sub_batch_size = int(round(exact.sum().item()))

        floored = torch.maximum(torch.floor(exact), torch.tensor(float(self.min_size)))
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
        return floored


class StochasticRounding(RoundingStrategy):
    def __init__(self, generator: torch.Generator):
        super().__init__()
        self.generator = generator

    def expected(self, exact: torch.Tensor) -> torch.Tensor:
        # unbiased by construction (see round), so the expectation is the ideal share
        return exact

    def round(self, exact: torch.Tensor) -> torch.Tensor:
        """
        Floors every share, then distributes the leftover slots by drawing from
        the fractional remainders.

        No minimum, so nothing needs clawing back: floor(exact) can never exceed
        the target sum, leaving at most (n_pids - 1) slots to distribute. Keeps
        E[count] exactly equal to the ideal share for every pid, which matters
        for a loss that is nonlinear in the summed background — a systematic
        over- or under representation biases it, sampling noise does not (in
        expectation).
        """

        sub_batch_size = int(round(exact.sum().item()))
        counts = exact.floor().to(torch.int32)

        # redistribute left overs stochastically
        leftover = sub_batch_size - int(counts.sum())
        if leftover > 0:
            frac = exact - counts.to(exact.dtype)
            picks = torch.multinomial(
                frac,
                leftover,
                replacement=False,
                generator=self.generator,
            )
            counts[picks] += 1

        return counts


def init_strategy(sampler_config):
    cfg = sampler_config
    choice = cfg.sampler_strategy_choice
    if choice == "stochastic":
        return StochasticRounding(cfg.active_config.generator)
    elif (choice == "largest_remainder") or (choice == "old"):
        return LargestRemainderRounding(cfg.active_config.min_size)
