from dataclasses import dataclass, field
from typing import Callable

from bbTT.configs import (
    BinningConfig,
    DataConfig,
    DebugConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    RecordConfig,
    SamplerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from bbTT.monitoring.logger.logger import get_logger

logger_inst = get_logger("Configs")


@dataclass
class FullConfig:
    dataset_config: DataConfig = field(default_factory=DataConfig)
    model_building_config: ModelConfig = field(default_factory=ModelConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    binning_config: BinningConfig = field(default_factory=BinningConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss_config: LossConfig = field(default_factory=LossConfig)
    debug_config: DebugConfig = field(default_factory=DebugConfig)
    record_config: RecordConfig = field(default_factory=RecordConfig)
    sampler_config: SamplerConfig = field(default_factory=SamplerConfig)

    def _compatibility_rules(self) -> list[tuple[Callable[[], bool], str]]:
        """Each rule: (is_violated, message). is_violated() returning True raises."""
        return [
            (
                lambda: self.optimizer_config.optimizer_choice == "sam" and self.training_config.training_fn != "sam",
                f"When using Optimizer SAM, training_fn must be 'sam', currently: {self.training_config.training_fn!r}",
            ),
            (
                lambda: (
                    self.loss_config.loss_fn == "cross_entropy"
                    and self.training_config.training_fn != "cross_entropy"
                    and self.training_config.validation_fn != "cross_entropy"
                ),
                f"Cross Entropy Loss expects the correct choices of Loop, current choice is:"
                f"training -> {self.training_config.training_fn}"
                f"validation -> {self.training_config.validation_fn}",
            ),
            (
                lambda: (
                    self.loss_config.loss_fn == "signal_efficiency"
                    and ((self.training_config.training_fn != "signal_efficiency")
                        or (self.training_config.validation_fn != "signal_efficiency"))
                ),
                f"Signal Efficiency Loss expects the correct choices of Loop, current choice is:"
                f"training -> {self.training_config.training_fn}"
                f"validation -> {self.training_config.validation_fn}",
            ),
            (
                lambda: (
                    (self.loss_config.loss_fn == "cross_entropy")
                    and (self.model_building_config.use_last_activation is False)
                ),
                "Cross Entropy Loss expect Logits and has build in SoftMax:"
            ),
            (
                lambda: (
                    (self.loss_config.loss_fn == "signal_efficiency")
                    and (self.model_building_config.use_last_activation is True)
                    and (self.model_building_config.last_activation_fn.lower() != "softmax")
                ),
                "Signal Efficiency expect SoftMax output."
                f"Currently using '{self.model_building_config.last_activation_fn.lower()}' as last activation function.",
            ),

            # add more rules here, e.g.:
            # (
            #     lambda: self.binning_config.enable_binning and not self.model_building_config.enable_binning,
            #     "BinningConfig.enable_binning=True requires ModelConfig.enable_binning=True",
            # ),
        ]

    def __post_init__(self):
        num_violations = 0

        for rule, message in self._compatibility_rules():
            if rule():
                num_violations += 1
                logger_inst.warning(f"{num_violations}:{message}")

        if num_violations:
            raise ValueError("Any rule is broken, see log above and fix rule before continue.")
