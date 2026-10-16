from __future__ import annotations

from typing import Optional

import torch
import torch.utils.data as t_data

from bbTT.data_handling.sampling.weight import WeightStatistics
from bbTT.monitoring.logger.logger import get_logger
from bbTT.utils.utils import CPU_DEVICE

logger_inst = get_logger(__name__)


class Process(t_data.Dataset):
    """
    Container for all data belonging to one process (process_id, process_type).
    Knows its own data/weights/statistics and how to gather a batch from itself.
    Holds NO cursor/iteration state — that lives in ProcessSampleCursor. The only mutable
    value is ``sample_size``, which ProcessSampler rewrites before every batch.
    """

    def __init__(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        target: torch.Tensor,
        normalization_weights: torch.Tensor,
        product_of_weights: torch.Tensor,
        weights_statistics: WeightStatistics,
        evaluation_space_mask: torch.Tensor,
        process_id: str = None,
        process_type: str = None,
    ):
        """
        A Process represents all data connected to a process identified by *process_id*,
        which itself belongs to a *process_type* (e.g. "tt", "dy", "hh").

        Args:
            continuous: Tensor of continuous features, shape [num_events, num_continuous_features]
            categorical: Tensor of categorical features, shape [num_events, num_categorical_features]
            target: One-hot target tensor, shape [num_events, num_classes]
            normalization_weights: Per-event normalization weights (MC oversampling correction), shape [num_events]
            product_of_weights: Per-event product of generation/reconstruction weights, shape [num_events]
            weights_statistics: Aggregated weight statistics for this process (see WeightAggregator)
            evaluation_space_mask: Bool mask selecting events in the evaluation phase space
            process_id: Unique id of the process
            process_type: Physics process type this process belongs to (e.g. "hh")
        """
        self.continuous = continuous.to(torch.float32)
        self.categorical = categorical.to(torch.int32)
        self.targets = target.to(torch.float32)

        # product of all weights assigned during generation and reconstruction;
        # necessary to transfer mc to real yield (real events one would see in data)
        self.product_of_weights = product_of_weights.to(torch.float32)

        # mc samples are generated with higher lumi (to reduce stat. unc.);
        # normalization weights scale down to correct lumi
        self.normalization_weights = normalization_weights.to(torch.float32)

        self.weights_statistics = weights_statistics
        self.evaluation_space_mask = evaluation_space_mask.to(torch.bool)

        self.process_id = process_id
        self.process_type = process_type
        self.sample_size = len(self)

        # cross-process bookkeeping set by ProcessSampler.calculate_sample_size, defaults are placeholders:
        #    sample_size: events this process contributes to the NEXT batch, redrawn per batch for stochastic rounding
        #    expected_sample_size: expectation of sample_size under the rounding strategy, fixed for a run
        #    relative_weight: ideal share of the batch before rounding, fixed for a run
        #    post_relative_weight: expected share of the batch under the rounding strategy, fixed for a run
        self.sample_size: int = len(self)
        self.expected_sample_size: float = float(len(self))
        self.relative_weight: Optional[float] = 0
        self.post_relative_weight: Optional[float] = 0

    @property
    def uid(self) -> tuple[str, str]:
        return (self.process_type, self.process_id)

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, idx):
        return self.continuous[idx], self.categorical[idx], self.targets[idx]

    def per_event_sample_weight(self, n: int, denominator: int ) -> torch.Tensor:
        """
        Single source of truth for the sample_weights value, used by sample/peek/generator alike.

        The weight is chosen such that the sum over the delivered events equals
        ``normalization_whole_sum``. The caller therefore has to pass how many events the
        delivered set contains in total:
            - the process' ``sample_size`` for a batch,
            - ``len(process)`` for a full pass.

        Using ``sample_size`` for a full pass over a training process would
        scale the weights by ``len(process) / sample_size``.

        Args:
            n (int): Number of events in this chunk, i.e. the length of the returned tensor.
            denominator (int): Total number of events the delivered set consists of.

        Returns:
            torch.Tensor: Shape (n, 1), every entry ``normalization_whole_sum / denominator``.
        """

        return torch.full(
            (n, 1),
            self.weights_statistics.normalization_whole_sum / denominator,
        )

    def per_event_composition_weight(self, n: int, denominator: int) -> torch.Tensor:
        """
        Per-event weight that reproduces the training batch composition on an unsampled set.

        In a training batch this process contributes the share ``post_relative_weight``
        of all events. On a full pass it contributes ``denominator`` events instead, so
        every event gets ``post_relative_weight / denominator``: summed over the delivered
        set the process then carries exactly the weight it has in a batch, which makes a
        loss over the full set comparable to the (unweighted) batch loss.

        Args:
            n (int): Number of events in this chunk, i.e. the length of the returned tensor.
            denominator (int): Total number of events the delivered set consists of, ``len(process)`` for a full pass.

        Returns:
            torch.Tensor: Shape (n, 1), every entry ``post_relative_weight / denominator``.
        """
        return torch.full(
            (n, 1),
            self.post_relative_weight / denominator,
        )



    def gather(
        self,
        idx: torch.Tensor,
        sample_from: tuple[str, ...],
        sample_weight_denominator: int,
        device=CPU_DEVICE
        ) -> dict[str, torch.Tensor]:
        """
        Pure lookup: given indices + attribute names, return the corresponding batch dict.
        No mutation, no cursor state — sample()/peek()/generator on ProcessSampleCursor all
        reduce to calling this.

        Args:
            idx (torch.Tensor): Indices to gather.
            sample_from (tuple[str, ...]): Attribute names to gather (e.g. "continuous", "categorical", "targets").
            sample_weight_denominator (int): Total size of the delivered set, see per_event_sample_weight.
            device (torch.device): Device to move tensors to.

        Returns:
            Dict of tensors for each requested attribute, plus "sample_weights".
        """
        out = {attribute: getattr(self, attribute)[idx].to(device) for attribute in sample_from}
        out["sample_weights"] = self.per_event_sample_weight(len(idx), denominator=sample_weight_denominator).to(device)
        return out


class ProcessSampleCursor:
    """
    Owns the mutable iteration state (current position, shuffle order) for repeatedly
    sampling from one Process.
    Kept separate from Process so:
    - Process stays a stateless, shareable data container
    - multiple independent cursors can sample the same Process without interfering
        (e.g. a train cursor and a monitoring/peek cursor over the same data)

    Each cursor owns a private ``torch.Generator`` seeded with *seed*.
    For checkpoints generator states are stored and loaded.
    This makes it 100% reproducible when reloading cursor states, since their generator has no noise.
    """

    def __init__(self, process: Process, randomize: bool = True, seed: int = 0):
        """
        Args:
            process (Process): Data container to iterate over.
            randomize (bool): Shuffle indices on every reset. If False, indices are in order.
            seed (int): Seed for the cursor's private generator. Use a different seed per cursor,
                otherwise all cursors produce identical permutations (see ProcessSampler).
        """
        self.process = process
        self.randomize = randomize
        self.seed = seed
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)
        # generator state before the permutation in self.indices was drawn, see load_state_dict
        self._state_before_reset: torch.Tensor = self.rng.get_state()
        self.reset()

    def reset(self):
        """Reset the cursor to the start, reshuffling indices if randomize=True."""
        self.current_idx = 0
        self.last_idx = None
        n = len(self.process)

        # remember the state first, so the same permutation can be drawn again on load
        self._state_before_reset = self.rng.get_state()
        self.indices = torch.randperm(n, generator=self.rng) if self.randomize else torch.arange(n)

    def state_dict(self) -> dict[str, int | torch.Tensor]:
        """
        Minimal description of the iteration state. Together with the process data this
        allows to continue sampling at exactly the same position.

        Returns:
            dict[str, int | torch.Tensor]: ``seed`` (for a consistency check), ``generator_state``
                (state before the current permutation was drawn) and ``current_idx``.
        """
        return {
            "seed": self.seed,
            "generator_state": self._state_before_reset.clone(),
            "current_idx": self.current_idx,
        }

    def load_state_dict(self, state: dict[str, int | torch.Tensor]) -> None:
        """
        Restore the iteration state produced by ``state_dict``. The current permutation is
        drawn again from the stored generator state, so the cursor continues with the same
        indices it would have produced originally. ``last_idx`` is not restored, a ``peek``
        directly after loading raises until the next ``sample``.

        Args:
            state (dict[str, int | torch.Tensor]): Output of ``state_dict`` of a cursor over the same process.

        Raises:
            ValueError: If the stored seed differs from the cursor's seed. The state would still
            load, but the mismatch signals that the sampler was built with another config.
        """
        if state["seed"] != self.seed:
            raise ValueError(
                f"Cursor seed mismatch for {self.process.uid}: stored {state['seed']}, cursor has {self.seed}."
            )
        self.rng.set_state(state["generator_state"])
        self.reset()
        self.current_idx = state["current_idx"]

    def sample_wrong(
        self, sample_from: tuple[str, ...], number: int = None, device=CPU_DEVICE
    ) -> dict[str, torch.Tensor]:
        """
        Sample *number* events from the process. Wraps around (reshuffling if randomize=True)
        once the end is reached. If *number* is None, the process's own sample_size is used.

        Args:
            sample_from: Attribute names to sample.
            number: Number of events to sample; defaults to process.sample_size.
            device: Device to move tensors to.

        Returns:
            Dict of sampled tensors, keyed by attribute name, plus "sample_weights".
        """
        num = len(self.process)
        if self.current_idx >= num:
            self.reset()

        number = self.process.sample_size if number is None else min(number, num)
        next_idx = min(self.current_idx + number, num)
        idx = self.indices[self.current_idx : next_idx]

        self.last_idx = idx
        self.current_idx = next_idx
        return self.process.gather(
            idx=idx,
            sample_from=sample_from,
            device=device,
            sample_weight_denominator=self.process.sample_size,
            )

    def sample(
        self,
        sample_from: tuple[str, ...],
        number: int = None,
        device=CPU_DEVICE,
    ) -> dict[str, torch.Tensor]:
        """
        Sample *number* events from the process.

        Always returns exactly *number* events. When the cursor reaches the end of
        the process it wraps around (reshuffling first if randomize=True) and keeps
        filling from the start, so the per-process quota in a batch is never
        silently truncated. Note that if *number* exceeds the process size, the
        returned batch necessarily contains repeated events.

        Args:

        sample_from (tuple[str]): Attribute names to sample.
        number (int, optional): Number of events to sample. Defaults to ``process.sample_size``.
        device (torch.device, optional): Device to move the gathered tensors to.

        Returns (dict[str, torch.Tensor]): Sampled tensors keyed by attribute name, plus ``"sample_weights"``.
        """
        max_events = len(self.process)
        remaining = self.process.sample_size if number is None else number
        if remaining < 0:
            raise ValueError(f"Number of events to sample must be positive, got {remaining}.")

        # Accumulate index chunks across wraparounds.
        # A loop is used to cover multiple spans
        chunks: list[torch.Tensor] = []
        while remaining > 0:
            if self.current_idx >= max_events:
                self.reset()
            take = min(remaining, max_events - self.current_idx)
            chunks.append(self.indices[self.current_idx : self.current_idx + take])
            self.current_idx += take
            remaining -= take

        # number == 0 --> process sample_size == 0, or user requested 0 events. In that case, chunks is empty.
        # In that case, return an empty tensor of the correct shape.
        idx = torch.cat(chunks) if chunks else self.indices[:0]

        self.last_idx = idx
        return self.process.gather(
            idx=idx,
            sample_from=sample_from,
            device=device,
            sample_weight_denominator=self.process.sample_size,
            )

    def peek(self, sample_from: tuple[str, ...], device=CPU_DEVICE) -> dict[str, torch.Tensor]:
        """Return the last sampled batch again, without advancing the cursor."""
        if self.last_idx is None:
            raise RuntimeError("No batch has been sampled yet. Call sample() before peek().")
        return self.process.gather(
            idx=self.last_idx,
            sample_from=sample_from,
            device=device,
            sample_weight_denominator=self.process.sample_size,
            )

    def generator(self, sample_from: tuple[str, ...], batch_size: int = -1, device=CPU_DEVICE):
        """
        Iterate over the whole process once, in order (no shuffling), independent of
        sample()/peek() cursor state. If batch_size == -1, everything is returned in one batch.

        Args:
            sample_from: Attribute names to gather per batch.
            batch_size: Events per yielded batch; -1 for the whole process at once.
            device: Device to move tensors to.

        Yields:
            Dict of tensors for each requested attribute, plus "sample_weights".
        """
        n = len(self.process)
        batch_size = n if batch_size == -1 else batch_size
        bounds = [(s, min(s + batch_size, n)) for s in range(0, n, batch_size)]

        # full pass: the delivered set is the whole process, not one batch
        for start, end in bounds:
            yield self.process.gather(
                idx=torch.arange(start, end),
                sample_from=sample_from,
                device=device,
                sample_weight_denominator=n,
                )
