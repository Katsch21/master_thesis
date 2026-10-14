from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from bbTT.configs.utils import choice_check

LOSS_CHOICE = Literal["cross_entropy", "signal_efficiency"]
SIGNAL_EFFICIENCY_LOSS_MODE = Literal["full", "no_unc", "approximation"]


@dataclass
class SignalEfficiencyLossConfig:
    asimov_mode: SIGNAL_EFFICIENCY_LOSS_MODE = "approximation"
    epsilon_small_signal: float = 1e-9
    epsilon_sqrt: float = 0.0
    epsilon_log: float = 1e-9
    background_uncertainty: float = 0.0


@dataclass
class WeightedCrossEntropyConfig:
    weight: float = 1.0
    reduction: str = "mean"
    label_smoothing: float = 0.0


@dataclass
class LossConfig:
    loss_fn: LOSS_CHOICE = "cross_entropy"
    signal_efficiency: SignalEfficiencyLossConfig = field(default_factory=SignalEfficiencyLossConfig)
    weighted_cross_entropy: WeightedCrossEntropyConfig = field(default_factory=WeightedCrossEntropyConfig)

    @property
    def active_config(self):
        return {
            "signal_efficiency": self.signal_efficiency,
            "cross_entropy": self.weighted_cross_entropy,
        }[self.loss_fn]

    def __post_init__(self):
        choice_check(self.loss_fn, LOSS_CHOICE)
        choice_check(self.signal_efficiency.asimov_mode, SIGNAL_EFFICIENCY_LOSS_MODE)
