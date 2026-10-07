from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from bbTT.configs.utils import choice_check

MODEL_CHOICE = Literal["residual", "dense", "lbn_dense", "binned_lbn_dense"]
TRAINING_LOOP_CHOICE = Literal["cross_entropy", "sam", "signal_efficiency"]
VALIDATION_LOOP_CHOICE = Literal["signal_efficiency", "cross_entropy"]
SAMPLING_STRATEGY = Literal["largest_remainder", "stochastic"]


@dataclass
class TrainingConfig:
    save_model_name: str = "baseline_baseline"  # name of the model used to save
    model_choice: MODEL_CHOICE = "lbn_dense"
    training_fn: TRAINING_LOOP_CHOICE = "cross_entropy"  # name of the training loop
    validation_fn: VALIDATION_LOOP_CHOICE = "cross_entropy"  # name of the validation loop
    max_train_iteration: int = 500_000  # max number of batches
    label_smoothing: float = 0.0
    train_folds: tuple[int, ...] = (0,)  # which training folds to use
    k_fold: int = 5
    seed: int = 100  # set torch and numpy seed for reproducibility
    train_ratio: float = 0.75  # split ratio for k-fold data into train and validation
    training_batch_size: int = int(4096 / 2) # -1 = full set
    validation_batch_size: int = 4096 * 20  # -1 = full set. -1 is discouraged, due to memory constraints # noqa

    # Sampler Settings

    def __post_init__(self):
        choice_check(self.training_fn, TRAINING_LOOP_CHOICE)
        choice_check(self.validation_fn, VALIDATION_LOOP_CHOICE)
        choice_check(self.model_choice, MODEL_CHOICE)
