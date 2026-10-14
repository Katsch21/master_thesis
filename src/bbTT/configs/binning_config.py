from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional

from bbTT.configs.utils import choice_check
from bbTT.utils.transformations import cubic, linspace, logit, tangent

BINNING_CHOICE = Literal["logit", "tangent", "linear", "cubic"]
KERNEL_CHOICE = Literal["Tanh"] # Gaussian Kernel and Linear are unfinished

# --- Binning functions and their corresponding configs


@dataclass
class LinearConfig:
    pass


@dataclass
class LogitConfig:
    eps: float = 1e-6


@dataclass
class TangentConfig:
    shift: float = 0.5


@dataclass
class CubicConfig:
    shift: float = 0.5
    min: float | None = None
    max: float | None = None
    stretching_factor: float | None = None


BINNING_REGISTRY: dict[BINNING_CHOICE, tuple[Any, Any]] = {
    "logit": (LogitConfig, logit),
    "tangent": (TangentConfig, tangent),
    "linear": (LinearConfig, linspace),
    "cubic": (CubicConfig, cubic),
}

# Kernel functions and their corresponding configs


@dataclass
class BinningConfig:
    num_bins: int = 10
    bounds: tuple[int] = (0, 1)
    # binning_fn: Any = torch.linspace  # keep as a callable, if needed
    binning_choice: BINNING_CHOICE = "logit"
    binning_cfg: Optional[None] = field(init=False)
    binning_fn: Optional[None] = field(init=False)

    # --- Determine the Kernel Configuration that is used for the ove
    kernel_cls: KERNEL_CHOICE = "Tanh"
    kernel_config: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "General": {
                "left_notch": 0.2,  # shift transition function towards bin center from left, cannot exceed half of bin width
                "right_notch": 0.2,  # shift transition function towards bin center from right
                "bin_height": 1,  # bin height should stay 1 normally
                "absolute_notch": False,  # notches are
            },
            "Tanh": {
                "eps": 1e-3,  # needs to be by default between 0 and 1
                "full_width": None,  # full width in transformed space where function goes from 1-eps to eps
            },
        }
    )

    def __post_init__(self):
        choice_check(self.kernel_cls, KERNEL_CHOICE)
        # extract configs and combine kernel configs and populate corresponding fields
        # kernel cfg
        kernel_general_config = self.kernel_config["General"].copy()
        kernel_choice_config = self.kernel_config[self.kernel_cls].copy()
        kernel_choice_config.update(kernel_general_config)
        self.kernel_config = kernel_choice_config

        # binning cfg
        binning_cfg, binning_fn = BINNING_REGISTRY[self.binning_choice]
        binning_cfg = asdict(binning_cfg()).copy()
        self.binning_fn = binning_fn
        self.binning_cfg = binning_cfg
