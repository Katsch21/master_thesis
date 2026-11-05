from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from bbTT.configs.utils import choice_check

OPTIMIZER_CHOICE = Literal["adamw"]


@dataclass
class ADAMWConfig:
    # weight decay lambda as in Loshchilov & Hutter (AdamW): per-step shrink at full learning rate
    weight_decay: float = 1.83e-6
    decouple_weight_decay_from_lr = True  # if true, decoupled weight decay with lr, as in AdamW

@dataclass
class OptimizerConfig:
    lr: float = 1e-2  # start learning rate
    optimizer_choice: OPTIMIZER_CHOICE = "adamw"

    def __post_init__(self):
        choice_check(self.optimizer_choice, OPTIMIZER_CHOICE)

    @property
    def active_config(self):
        return {
            "adamw": ADAMWConfig,
        }[self.optimizer_choice]
