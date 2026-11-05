from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RecordConfig:
    verbose_interval: int = 5  # interval between two logger training status print outs
    validation_interval: int = 200  # interval between two validation passes / plots are done during validation

    # Checkpoint settings, evaluated once per validation pass
    # A checkpoint is written when the validation loss beats the best confirmed loss by more than `checkpoint_delta`.
    # The best confirmed loss is also the metric a ReduceLROnPlateau scheduler sees (see SchedulerHandler), so this
    # is the finer of the two thresholds: the scheduler's `threshold` decides on top of it what still counts as
    # progress. Keep checkpoint_delta <= scheduler threshold, otherwise checkpoints stop before the scheduler reacts.
    checkpoint_delta: float = 0.0  # minimum improvement of the validation loss to count as new best
    checkpoint_patience: int = 0  # number of validations to skip after a written checkpoint before looking again
    # Hold a new best as candidate and write it only if the next validation also beats the previous best.
    # Protects checkpoint and scheduler from a single validation spike; costs one validation_interval of delay.
    checkpoint_confirm: bool = True


    # --- Monitoring settings ---
    batch_metrics: bool = False  # if batch metrics should be tracked, this is only for debugging and not used in production
    log_metrics: bool = True  # log metrics to tensorboard during training
<<<<<<< HEAD

    # full pass over the training data is pure monitoring but costs ~4x a validation pass.
    training_evaluation_interval: int = 5 # 1 = every validation pass, 2 = every second etc.

    # what plots and scalar should be monitored during training, validation and batch monitoring.
    # PLOTS are defined in PLOT_REGISTRY.
    # SCALAR map a tensorboard "tag/name" to an artifact the context can
    batch_plots: tuple[str, ...] = (
        # "kernels_monitor",
        "batch_composition_history",
    )
    training_plots: tuple[str, ...] = (
        "confusion_matrix",
        # "roc",  # roc computation is the long factor (14s)
        "output_score_hh_node",
        "output_score_hh_node_physics",

        "output_score_hh_node_untransformed",
        "kernels_monitor",
        "score_correlation_matrix",
        # "precision_recall",  # deactivated takes 40s to use
    )
    validation_plots: tuple[str, ...] = (
        "confusion_matrix",
        # "roc",  # roc computation is the long factor (4s)
        "output_score_hh_node",
        "output_score_hh_node_physics",

        "output_score_hh_node_untransformed",
        "kernels_monitor",
        "score_correlation_matrix",
        # "precision_recall",  # deactivated takes 12s to use
    )
    # needs to be dict due to how tensorboard handles scalars
    training_scalars: dict[str, str] = field(
        default_factory=lambda: {
            "CrossEntropy/Evaluation Training": "cross_entropy",
            "CrossEntropy per class - Training": "cross_entropy_per_class",
            "Loss/Evaluation Training Loss": "loss",
        }
    )
    validation_scalars: dict[str, str] = field(
        default_factory=lambda: {
            "CrossEntropy/Evaluation Validation": "cross_entropy",
            "CrossEntropy per class - Validation": "cross_entropy_per_class",
            "Loss/Validation VLoss": "loss",
        }
    )
=======
    verbose_interval: int = 5  # interval between two logger training status print outs
    validation_interval: int = 500  # interval between two validation passes / plots are done during validation
>>>>>>> 86f6fbf (changed validation interval to a much larger value (from 30 to 500), because batch size is so much smaller now and validation should not happen too often)
