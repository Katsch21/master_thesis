from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RecordConfig:
    batch_metrics: bool = False  # if batch metrics should be tracked, this is only for debugging and not used in production
    log_metrics: bool = True  # log metrics to tensorboard during training
    verbose_interval: int = 5  # interval between two logger training status print outs
    validation_interval: int = 500  # interval between two validation passes / plots are done during validation
