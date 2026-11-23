from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

from bbTT.configs.utils import choice_check

SCHEDULER_CHOICE = Literal["cosine_annealing", "half_cosine", "reduce_on_plateau", "step", "exponential"] # main scheduler choices

WARMUP_CHOICE = Literal["linear", "cosine"] # warm-up curves, implemented in bbTT.optimization.scheduler.warmup.WARMUP_REGISTRY

@dataclass
class StepLRConfig:  # used by marcel
    step_size: int = 8000  # number of iterations between two learning rate reductions
    gamma: float = 0.5  # learning rate reduction factor


@dataclass
class CosineAnnealingLRConfig:
    T_max: int = 5000  # maximum number of iterations
    eta_min: float = 1e-5  # minimum learning rate

@dataclass
class HalfCosineLRConfig:
    # modification of cosine annealing, cosine decay that stays at eta_min instead of rising again
    T_max: int = 5000  # iterations of the half wave, counted from the end of the warm-up
    eta_min: float = 1e-9  # absolute learning rate reached after T_max and kept from then on


@dataclass
class ReduceLROnPlateauConfig:
    # The metric this scheduler sees is NOT the raw validation loss but the checkpoint's best confirmed loss
    # (see SchedulerHandler and CheckpointConfig). It only changes when a checkpoint is confirmed, so `patience`
    # counts validations without a confirmed improvement and `threshold` is applied on top of CheckpointConfig.delta.
    mode: str = "min"  # min or max. min = lr reduces when the quantity monitored has stopped decreasing
    patience: int = 4  # wait x number of checks - Marcel (old setup): 10, with small batch size: must be much smaller now!
    threshold_mode: str = "abs"  # type of min_delta - Marcel: abs
    factor: float = 0.7  # LR reduce by factor
    cooldown: int = 0  # number of iterations to wait after a learning rate reduction before resuming normal operation
    min_lr: float = 1e-6  # lower bound on the learning rate
    eps: float = 1e-08  # minimal decay applied to lr, if it is smaller than this value, it is set to this value
    threshold: float = 1e-4  # minimum change of the confirmed best loss to count as improvement (torch default)


@dataclass
class ExponentialLRConfig:
    gamma: float = 0.9  # Multiplicative factor of learning rate decay.
    last_epoch: int = -1  # The index of last epoch. Default: -1.


SCHEDULER_REGISTRY = {
    "cosine_annealing": (CosineAnnealingLRConfig, "CosineAnnealingLR"),
    "half_cosine": (HalfCosineLRConfig, "HalfCosineLR"),
    "reduce_on_plateau": (ReduceLROnPlateauConfig, "ReduceLROnPlateau"),
    "step": (StepLRConfig, "StepLR"),
    "exponential": (ExponentialLRConfig, "ExponentialLR"),
}


@dataclass
class SchedulerConfig:
    # --- Main Config ---
    # main scheduler, takes over after the warm-up (or from the start without one)
    scheduler_choice: SCHEDULER_CHOICE = "reduce_on_plateau"

    # --- WarmUp Config ---
    # warm-up curve, None disables the warm-up. Optional[...] instead of "| None": from_dict evaluates the hints
    warmup_choice: Optional[WARMUP_CHOICE] = "cosine"
    warmup_iterations: int = 100  # batch steps until the warm-up factor reaches 1 (= base learning rate)
    warmup_start_factor: float = 1e-4  # factor of the base learning rate at the first iteration

    # --- Main scheduler settings ---
    # one sub config per SCHEDULER_CHOICE, the field name equals the choice. Only the chosen one is used,
    # but all are fields so they are overridable (--override scheduler_config.half_cosine.T_max=19900)
    # and saved with the checkpoint.
    cosine_annealing: CosineAnnealingLRConfig = field(default_factory=CosineAnnealingLRConfig)
    half_cosine: HalfCosineLRConfig = field(default_factory=HalfCosineLRConfig)
    reduce_on_plateau: ReduceLROnPlateauConfig = field(default_factory=ReduceLROnPlateauConfig)
    step: StepLRConfig = field(default_factory=StepLRConfig)
    exponential: ExponentialLRConfig = field(default_factory=ExponentialLRConfig)

    @property
    def active_config(self) -> Any:
        """
        Returns:
            Any: Sub config of the chosen main scheduler. Checkpoints written before the sub configs were
                fields lack the attribute (from_dict skips __init__), then the code default is used.
        """
        default_cls = SCHEDULER_REGISTRY[self.scheduler_choice][0]
        return getattr(self, self.scheduler_choice, None) or default_cls()

    @property
    def scheduler_kwargs(self) -> dict[str, Any]:
        """
        Returns:
            dict[str, Any]: Keyword arguments of the main scheduler, taken from ``active_config``.
        """
        return asdict(self.active_config)

    @property
    def scheduler_cls(self) -> str:
        """
        Returns:
            str: Class name of the main scheduler, resolved to the class when the scheduler is built.
        """
        return SCHEDULER_REGISTRY[self.scheduler_choice][1]

    def __post_init__(self):
        choice_check(self.scheduler_choice, SCHEDULER_CHOICE)
        if self.warmup_choice is not None:
            choice_check(self.warmup_choice, WARMUP_CHOICE)
            if self.warmup_iterations <= 0:
                raise ValueError(f"warmup_iterations must be positive, got {self.warmup_iterations}")
