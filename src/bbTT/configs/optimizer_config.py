from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from bbTT.configs.utils import choice_check

OPTIMIZER_CHOICE = Literal["adamw", "sam"]


@dataclass
class ADAMWConfig:
    decay_factor: float = 500  # factor of L2 - Marcel: 500
    normalize: bool = True  # normalize weight decay factor to number of parameters
    lr: float = 3e-3  # start learning rate


@dataclass
class SAMConfig:
    TODO = None


@dataclass
class OptimizerConfig:
    optimizer_choice: OPTIMIZER_CHOICE = "adamw"

    def __post_init__(self):
        choice_check(self.optimizer_choice, OPTIMIZER_CHOICE)

    @property
    def active_config(self):
        return {
            "adamw": ADAMWConfig,
            "sam": SAMConfig,
        }[self.optimizer_choice]
