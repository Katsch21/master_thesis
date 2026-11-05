from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

from bbTT.configs.utils import choice_check

SCHEDULER_CHOICE = Literal["linear", "cosine_annealing", "reduce_on_plateau", "step", "exponential"]


@dataclass
class StepLRConfig:  # used by marcel
    step_size: int = 700  # number of iterations between two learning rate reductions
    gamma: float = 0.5  # learning rate reduction factor


@dataclass
class CosineAnnealingLRConfig:
    T_max: int = 5000  # maximum number of iterations
    eta_min: float = 1e-5  # minimum learning rate


@dataclass
class ReduceLROnPlateauConfig:
    mode: str = "min"  # min or max. min = lr reduces when the quantity monitored has stopped decreasing
    patience: int = 4  # wait x number of checks - Marcel (old setup): 10, with small batch size: must be much smaller now!
    threshold_mode: str = "abs"  # type of min_delta - Marcel: abs
    factor: float = 0.7  # LR reduce by factor
    cooldown: int = 0  # number of iterations to wait after a learning rate reduction before resuming normal operation
    min_lr: float = 1e-6  # lower bound on the learning rate
    eps: float = 1e-08  # minimal decay applied to lr, if it is smaller than this value, it is set to this value


@dataclass
class LinearLRConfig:
    start_factor: float = 1e-4  # the initial learning rate will be the start_factor times the base learning rate
    end_factor: float = 1.0  # the final learning rate will be the end_factor times the base learning rate
    total_iters: int = 200  # number of iterations over which the multiplier increases from start_factor to end_factor


@dataclass
class ExponentialLRConfig:
    gamma: float = 0.9  # Multiplicative factor of learning rate decay.
    last_epoch: int = -1  # The index of last epoch. Default: -1.


SCHEDULER_REGISTRY = {
    "cosine_annealing": (CosineAnnealingLRConfig, "CosineAnnealingLR"),
    "reduce_on_plateau": (ReduceLROnPlateauConfig, "ReduceLROnPlateau"),
    "linear": (LinearLRConfig, "LinearLR"),
    "step": (StepLRConfig, "StepLR"),
    "exponential": (ExponentialLRConfig, "ExponentialLR"),
}


@dataclass
class SchedulerConfig:
    scheduler_chain: tuple[SCHEDULER_CHOICE, ...] = ("reduce_on_plateau",) # schedulers used in chain
    # scheduler_chain: tuple[SCHEDULER_CHOICE, ...] = ("linear", "reduce_on_plateau")  # schedulers used in chain
    milestones: tuple[int, ...] = ()  # intervals after which the LR scheduler is swapped
    # milestones: tuple[int, ...] = (200,)  # intervals after which the LR scheduler is swapped

    config_chain: Optional[tuple[Any, ...]] = (
        None  # list of configs corresponding to the schedulers in the scheduler chain
    )
    scheduler_cls_chain: Optional[tuple[Any, ...]] = None # noqa list of scheduler classes corresponding to the schedulers in the scheduler chain

    def __post_init__(self):
        scheduler_configs = []
        scheduler_cls_chain = []
        for choice in self.scheduler_chain:
            config, scheduler_cls = SCHEDULER_REGISTRY[choice]
            scheduler_configs.append(asdict(config()))
            scheduler_cls_chain.append(scheduler_cls)  # init is done in the training
            choice_check(choice, SCHEDULER_CHOICE)
        self.config_chain = tuple(scheduler_configs)
        self.scheduler_cls_chain = tuple(scheduler_cls_chain)
