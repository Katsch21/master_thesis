from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from bbTT.configs.utils import choice_check

MODEL_CHOICE = Literal["residual", "dense", "lbn_dense", "binned_lbn_dense"]
SAMPLING_STRATEGY = Literal["largest_remainder", "stochastic"]


@dataclass
class TrainingConfig:
    model_choice: MODEL_CHOICE = "lbn_dense"
    max_train_iteration: int = 20_000  # max number of batches
    label_smoothing: float = 0.0
    train_folds: tuple[int, ...] = (0,)  # which training folds to use
    k_fold: int = 5
    seed: int = 100  # set torch and numpy seed for reproducibility
    train_ratio: float = 0.75  # split ratio for k-fold data into train and validation
    randomize: bool = True  # randomize the data before splitting into train and validation - this option is important for evaluation, otherwise training and validation data are not independent. # noqa
    training_batch_size: int = int(4096 / 2) # -1 = full set
    validation_batch_size: int = 4096 * 30  # -1 = full set. -1 is discouraged, due to memory constraints # noqa

    # forward passes without optimizer step that fill the BatchNorm running statistics with real data before
    # the first iteration, see train_utils.calibrate_batch_norms. 0 disables it. Skipped on resume.
    batch_norm_calibration_batches: int = 50
    def __post_init__(self):
        choice_check(self.model_choice, MODEL_CHOICE)
