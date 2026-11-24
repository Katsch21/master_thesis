from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from bbTT.configs.utils import choice_check

OPTIMIZER_CHOICE = Literal["adamw"]


@dataclass
class ADAMWConfig:
    # weight decay lambda as in Loshchilov & Hutter (AdamW): per-step shrink at full learning rate
    weight_decay: float = 1.83e-5
    decouple_weight_decay_from_lr: bool = False  # if true, decoupled weight decay with lr, as in AdamW


@dataclass
class OptimizerConfig:
    lr: float = 3e-3  # start learning rate
    optimizer_choice: OPTIMIZER_CHOICE = "adamw"

    # one sub config per OPTIMIZER_CHOICE, field name equals the choice, so it is overridable
    # (--override optimizer_config.adamw.weight_decay=1e-4) and saved with the checkpoint
    adamw: ADAMWConfig = field(default_factory=ADAMWConfig)



    def __post_init__(self):
        choice_check(self.optimizer_choice, OPTIMIZER_CHOICE)

    @property
    def active_config(self) -> ADAMWConfig:
        """
        Returns:
            ADAMWConfig: Sub config of the chosen optimizer. Checkpoints written before the sub config was a
                field lack the attribute (from_dict skips __init__), then the code default is used.
        """
        return getattr(self, self.optimizer_choice, None) or ADAMWConfig()
