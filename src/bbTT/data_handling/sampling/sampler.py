from __future__ import annotations

import zlib
from collections import defaultdict
from typing import TYPE_CHECKING, Iterable, Optional

import torch
import torch.utils.data as t_data

from bbTT.data_handling.sampling.batch_size_calculation import BatchSizeAllocator
from bbTT.data_handling.sampling.process import Process, ProcessSampleCursor
from bbTT.data_handling.sampling.sample_strategy import init_strategy
from bbTT.monitoring.logger.logger import get_logger
from bbTT.utils.utils import CPU_DEVICE

if TYPE_CHECKING:
    from bbTT.configs.full_config import FullConfig

logger_inst = get_logger(__name__)


class ProcessRegistry:
    """
    Flat storage + lookup for Process instances, keyed by uid = (process_type, process_id).
    Single source of truth — replaces the old nested {type: {id: Process}} dict plus its
    flattened processes() view.
    """

    def __init__(self):
        self._processes: dict[tuple[str, str], Process] = {}

    def add(self, process: Process):
        self._processes[process.uid] = process

    def __getitem__(self, uid: tuple[str, str]) -> Process:
        return self._processes[uid]

    def __len__(self) -> int:
        return sum(len(p) for p in self._processes.values())

    def __contains__(self, uid: tuple[str, str]) -> bool:
        return uid in self._processes

    def all(self) -> dict[tuple[str, str], Process]:
        return dict(self._processes)

    def by_type(self, process_type: str) -> dict[str, Process]:
        """process_id -> Process, restricted to one process_type."""
        return {uid[1]: p for uid, p in self._processes.items() if uid[0] == process_type}

    def get_attribute_of_datasets(self, attr: str) -> dict[tuple[str, str], object]:
        """
        Generic accessor for ad hoc inspection/debugging — e.g. printing sample_size
        or relative_weight across all processes without writing a dedicated method.
        Prefer a typed method (like total_weight_by_type) for anything used in
        actual control flow.
        """
        return {uid: getattr(proc, attr) for uid, proc in self._processes.items()}

    @property
    def process_types(self) -> list[str]:
        return sorted({uid[0] for uid in self._processes})

    def total_weight_by_type(self, weight_attr: str) -> dict[str, float]:
        """
        Sum a weight-producing attribute/callable across processes, grouped by each
        Process's own process_type. Replaces the pid-range guessing that used to live
        in sampler_total_weight_per_process_type / sampler_total_eval_weight_per_process_type.

        Args:
            weight_attr: Name of a Process attribute/method returning a per-uid summable value.
        """
        totals: dict[str, float] = defaultdict(float)
        for uid, proc in self._processes.items():
            value = getattr(proc, weight_attr)
            value = value() if callable(value) else value
            totals[uid[0]] += float(value)
        return dict(totals)


class ProcessSampler(t_data.Sampler):
    """
    Manager for Process instances that regulates sampling across multiple processes.
    Each Process knows how to sample itself; ProcessSampler orchestrates composition
    of a combined batch across processes.

    Batch composition happens in two stages:
        1. Desired events per process_type, from sample_ratio (must sum to 1.0).
        2. Desired events per process_id within a process_type, proportional to that
        process's total normalization weight (via BatchSizeAllocator).
    """

    shared_attributes = {
        "relative_weight",
        "post_relative_weight",
        "expected_sample_size",
    }

    def __init__(
        self,
        batch_size: int = 1,
        sample_ratio: Optional[dict[str, float]] = None,
        sub_sample_ratio: Optional[dict[str, float]] = None,
        target_map: Optional[dict[str, int]] = None,
        seed: int = 0,
        weight_aggregator_inst=None,
        kind="training",
        bs_allocator=None,
    ):
        """
        Args:
            batch_size: Number of events per sampled batch.
            sample_ratio: process_type -> fraction of the batch. Defaults to
                {"dy": 0.25, "tt": 0.25, "hh": 0.5}.
            sub_sample_ratio: process_id -> relative importance multiplier within its type.
            min_size: Minimum number of events per process_id.
            target_map: process_type -> index in the one-hot target tensor. Defaults to
                {"hh": 0, "tt": 1, "dy": 2}.
            weight_aggregator_inst: Reference to the WeightAggregator that computed the
                weight statistics fed into this sampler's Processes. Not called directly
                by ProcessSampler — kept as a passive reference for downstream consumers.
        """
        self.registry = ProcessRegistry()
        self.cursors: dict[tuple[str, str], ProcessSampleCursor] = {}
        self.batch_size = batch_size
        self.sample_ratio = sample_ratio or {"dy": 0.25, "tt": 0.25, "hh": 0.5}
        self.sub_sample_ratio = sub_sample_ratio or {}
        self.target_map = target_map or {"hh": 0, "tt": 1, "dy": 2}
        self.weights_aggregator_inst = weight_aggregator_inst
        # self.allocator = BatchSizeAllocator(min_size=min_size)
        self.allocator = bs_allocator
        self.kind = kind
        self.seed = seed

    def cursor_seed(self, uid: tuple[str, str]) -> int:
        """
        Derive a stable per-process seed from the sampler's base seed and the process uid.

        ``zlib.crc32`` is used instead of ``hash`` because string hashing is randomized per
        python process (PYTHONHASHSEED); the seed must be identical across runs, otherwise a
        restored cursor state would not match the cursor it is loaded into.

        Args:
            uid (tuple[str, str]): (process_type, process_id) of the process.

        Returns:
            int: Seed for the cursor of this process.
        """
        return self.seed + zlib.crc32(repr(uid).encode())

    def add_process_instance(self, process: Process, randomize: bool = False):
        """Register a Process with the sampler."""
        self.registry.add(process)
        # each process gets a unique seed by the sampler to decouple their random generators
        self.cursors[process.uid] = ProcessSampleCursor(process, randomize=randomize, seed=self.cursor_seed(process.uid))

    def state_dict(self) -> dict[tuple[str, str], dict]:
        """
        Iteration state of every registered cursor, keyed by uid. Only plain data
        (ints and tensors) ends up in here, so the result can go straight into torch.save.

        Returns:
            dict[tuple[str, str], dict]: uid -> cursor state, see ProcessSampleCursor.state_dict.
        """
        return {uid: cursor.state_dict() for uid, cursor in self.cursors.items()}

    def load_state_dict(self, state: dict[tuple[str, str], dict]) -> None:
        """
        Restore every cursor from a state produced by state_dict. The set of registered
        processes must match exactly: a cursor without state would silently continue at
        its own position and break the "same batches again" guarantee.

        Args:
            state (dict[tuple[str, str], dict]): Output of state_dict of a sampler over the same processes.

        Raises:
            ValueError: If the stored uids and the registered uids differ.
        """
        stored, registered = set(state), set(self.cursors)
        if stored != registered:
            raise ValueError(
                "Sampler state does not match registered processes.\n"
                f"Missing in state: {sorted(registered - stored)}\n"
                f"Unknown in state: {sorted(stored - registered)}"
            )
        for uid, cursor_state in state.items():
            self.cursors[uid].load_state_dict(cursor_state)

    def __contains__(self, uid: tuple[str, str]) -> bool:
        return uid in self.registry

    def __len__(self) -> int:
        return len(self.registry)

    def events_per_dataset(self) -> dict[tuple[str, str], int]:
        return {uid: len(p) for uid, p in self.registry.all().items()}

    def print_sample_rates(self):
        """Log the current sample_size of every registered process, grouped by process_type."""
        by_type: dict[str, list[str]] = defaultdict(list)
        for uid, proc in self.registry.all().items():
            process_type, pid = uid
            by_type[process_type].append(f"{pid} | {proc.sample_size}")

        msg = "\n".join(
            f"{process_type.upper()}:\n\t" + "\n\t".join(entries) for process_type, entries in by_type.items()
        )
        logger_inst.debug("Sample rates per process (PID | Sample Size)\n" + msg)

    def calculate_sample_size(self, process_type: str):
        """
        Draw the per-process event counts of the next batch for one process_type and
        write them into the processes. Called before every batch by ``sample_batch``.
        A stochastic rounding strategy gets a fresh draw each time.
        A deterministic strategy returns the same counts on every call.

        Args:
            process_type (str): Parent process whose sub-processes are allocated.

        Raises:
            ValueError: If the batch size is too small to be split.
        """
        if self.batch_size <= 1:
            raise ValueError("Batch size <= 1 is not supported. Try a number big enough to be representative.")

        procs_by_pid = self.registry.by_type(process_type)

        weights_by_pid = {
            pid: proc.weights_statistics.normalization_whole_sum.item() for pid, proc in procs_by_pid.items()
        }
        sub_batch_size = int(self.batch_size * self.sample_ratio[process_type])

        # allocate describes the actual algorithm, can be swapped out freely.
        sizes, expected_sizes, relative_weights, post_relative_weights = self.allocator.allocate(
            weights_by_pid=weights_by_pid,
            sub_sample_ratio=self.sub_sample_ratio,
            sub_batch_size=sub_batch_size,
            sample_ratio_for_type=self.sample_ratio[process_type],
        )
        for pid, size in sizes.items():
            procs_by_pid[pid].sample_size = size
        for pid, rel_w in relative_weights.items():
            procs_by_pid[pid].relative_weight = rel_w
        for pid, post_rel_w in post_relative_weights.items():
            procs_by_pid[pid].post_relative_weight = post_rel_w
        for pid, expected_size in expected_sizes.items():
            procs_by_pid[pid].expected_sample_size = expected_size

    def redraw_sample_sizes(self):
        """Redraw the batch composition of every process_type, see calculate_sample_size."""
        for process_type in self.registry.process_types:
            self.calculate_sample_size(process_type=process_type)

    def load_process_weights_from(self, other: "ProcessSampler"):
        """
        Copy attributes from every process in *other* into the matching process here
        (matched by uid). Other is another ProcessSampler, like Training.
        To define which attributes are shared, change the class attribute called 'shared_attributes'.
        """
        for uid, proc in self.registry.all().items():
            if uid in other:
                for shared_attribute in self.shared_attributes:
                    other_uid = other.registry[uid]
                    other_attr_value = getattr(other_uid, shared_attribute)
                    setattr(proc, shared_attribute, other_attr_value)

    # --- Sample Mechanism ---
    def sorted_pid_registry(self) -> Iterable[tuple[int, Process]]:
        """Small helper that returns an iterable of all registered processes, sorted by process id.

        Returns:
            Iterable[tuple[int, Process]]: Sorted Dictionary iterable, where process id is key and Process is value.
        """
        return sorted(self.registry.all(), key=lambda u: u[1])

    def _aggregate_batch(
        self,
        sample_from: list[str],
        cursor_method: str,
        device: torch.device = CPU_DEVICE,
    ) -> dict[str, torch.tensor]:
        """
        Shared aggregation logic for sample_batch/peek_batch: calls cursor_method
        (either "sample" or "peek") on every registered process's cursor, in pid order,
        and concatenates each attribute across processes.

        Args:
            sample_from: Attribute names to sample from each process (e.g.
                ["continuous", "categorical", "targets"]).
            cursor_method: Name of the ProcessSampleCursor method to call ("sample" or "peek").
            device: Device to move the final tensors to.
        Returns:
            dict[attribute, Tensor], keys from sample_from, values are concatenated across all processes.
        """
        events: dict[str, list[torch.Tensor]] = defaultdict(list)

        for uid in self.sorted_pid_registry():
            cursor = self.cursors[uid]
            method = getattr(cursor, cursor_method)
            for attribute, sampled in method(sample_from, device=device).items():
                events[attribute].append(sampled)

        return {attribute: torch.concatenate(tensors, dim=0).to(device) for attribute, tensors in events.items()}

    def sample_batch(self, sample_from: list[str], device: torch.device = CPU_DEVICE) -> dict[str, torch.Tensor]:
        """
        Redraw the batch composition, then sample ``sample_size`` events from every
        registered process and concatenate them.
        """
        self.redraw_sample_sizes()
        return self._aggregate_batch(cursor_method="sample", sample_from=sample_from, device=device)

    def peek_batch(self, sample_from: list[str], device: torch.device = CPU_DEVICE) -> dict[str, torch.Tensor]:
        """
        Reconstruct the last batch produced by sample_batch(), without advancing any
        cursor.

        Raises:
            RuntimeError: If sample_batch() hasn't been called yet for some process
                (propagated from ProcessSampleCursor.peek()).
        """
        return self._aggregate_batch(cursor_method="peek", sample_from=sample_from, device=device)

    def full_pass(self, sample_from: list[str], batch_size: int = -1, device: torch.device = CPU_DEVICE):
        """
        Walk through every registered process's full data exactly once, process by
        process, using each process's own generator(). Batch composition is deactivated
        for this method.

        Args:
            sample_from: Attribute names to gather.
            batch_size: Chunk size per generator step (-1 = whole process at once).
            device: Device to move tensors to.

        Yields:
            (uid, batch_dict) tuples — uid tells you which process the batch came from.
        """
        for uid in self.sorted_pid_registry():
            cursor = self.cursors[uid]
            for batch in cursor.generator(sample_from, batch_size, device):
                yield uid, batch

    def missing_processes(self) -> bool:
        """
        Checks if all requested sub-samples actually are registed.
        When mismatch exist stem a warning.

        Returns (bool): True, if any requested sample is mising.
        """

        requested_sub_samples = set(self.sub_sample_ratio.keys())
        # pid of all registered processes
        registered_processes = set([pid[1] for pid in self.registry.all().keys()])
        # check if requested process ideas exist in sampler
        missing = sorted(requested_sub_samples - registered_processes)
        if missing:
            logger_inst.warning(f"requested pids {missing} are not within the sampler and get dummy weight of 1")
        return bool(missing)


def create_sampler(
    events: dict, weight_aggregator_inst, full_config: FullConfig, train: bool = False
) -> ProcessSampler:
    """
    Build a ProcessSampler from raw per-uid event dicts, wrapping each into a Process
    and registering it. If train=True, also computes sample sizes for every process_type.

    Args:
        events: dict[uid, dict] with keys "continuous", "categorical", "normalization_weights", "product_of_weights", "evaluation_mask". Consumed (popped) during construction.
        weight_aggregator_inst: WeightAggregator whose .weights[uid] gives a WeightStatistics for each uid in events.
        full_config (FullConfig): FullConfig instance object. All necessary arguments a picked by the program itself.
        train (bool): Is a training or validation sampler created.
    Returns:
        A populated ProcessSampler.

    Raises:
        ValueError: If events is empty.
    """

    if not events:
        raise ValueError("Sampler is not created due to feeding empty events")

    batch_strategy = init_strategy(sampler_config=full_config.sampler_config)
    batch_size_allocator = BatchSizeAllocator(rounding=batch_strategy)

    kind = ("training" if train else "validation")

    process_sampler = ProcessSampler(
        batch_size=full_config.training_config.training_batch_size,
        sample_ratio=(full_config.sampler_config.sample_ratio or {"dy": 0.25, "tt": 0.25, "hh": 0.5}),
        target_map=full_config.dataset_config.target_map,
        weight_aggregator_inst=weight_aggregator_inst,
        sub_sample_ratio=full_config.sampler_config.sub_process_ratios,
        bs_allocator=batch_size_allocator,
        seed=full_config.training_config.seed,
        kind=kind,
    )

    for uid in list(events.keys()):
        process_type, process_id = uid

        logger_inst.debug(
            f"{kind}: Add {process_type} pid: {process_id}"
        )
        arrays = events.pop(uid)

        num_events = len(arrays["continuous"])
        target_value = full_config.dataset_config.target_map[process_type]
        num_of_targets = len(full_config.dataset_config.target_map)
        target = torch.zeros(size=(num_events, num_of_targets), dtype=torch.float32)
        target[:, target_value] = 1.0

        process = Process(
            continuous=arrays["continuous"],
            categorical=arrays["categorical"],
            target=target,
            normalization_weights=arrays["normalization_weights"],
            product_of_weights=arrays["product_of_weights"],
            weights_statistics=weight_aggregator_inst.weights[uid],
            evaluation_space_mask=arrays["evaluation_mask"],
            process_id=process_id,
            process_type=process_type,
        )
        process_sampler.add_process_instance(process, randomize=True)

    if train:
        process_sampler.missing_processes()

        for process_type in process_sampler.registry.process_types:
            process_sampler.calculate_sample_size(process_type=process_type)

    return process_sampler
