from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch

from data import features
from data.utils import find_datasets
from utils.utils import EMPTY_FLOAT, choice_check, multiply_sub_process_rates

# some values only accept certain values, not runtime check is performed!
ERAS_CHOICE = Literal["22pre", "22post", "23pre", "23post"]
LAST_ACTIVATION_CHOICE = Literal["Softmax", "Sigmoid", None]
KERNEL_CHOICE = Literal["GaussianKernelV3", "GaussianKernelFinal"]
OPTIMIZER_CHOICE = Literal["adamw", "sam"]

MODEL_CHOICE = Literal["residual", "dense", "lbn_dense", "binned_lbn_dense"]
LOSS_CHOICE = Literal["cross_entropy", "signal_efficiency", "signal_efficiency_binning_aware"]
TRAINING_LOOP_CHOICE = Literal["cross_entropy", "sam", "signal_efficiency"]
VALIDATION_LOOP_CHOICE = Literal["signal_efficiency", "cross_entropy"]
SIGNAL_EFFICIENCY_LOSS_MODE = Literal["full", "no_unc", "approximation"]
SCHEDULER_CHOICE = Literal["linear", "cosine_annealing", "reduce_on_plateau", "step"]



@dataclass
class DataConfig:
    # changes in this config will create a NEW hash of the data
    target_map: Dict[str, int] = field(default_factory=lambda: {"hh": 0, "tt": 1, "dy": 2}) # node: index
    continuous_features: List[str] = features.continuous_features
    categorical_features: List[str] = features.categorical_features
    dataset_pattern: Tuple[str] = (
        "dy_*",
        "tt_*",
        "hh_ggf_hbb_htt_kl1_kt1*",
        # "hh_ggf_hbb_htt_kl0_kt1*",
        )
    eras: Tuple[ERAS_CHOICE] = ("22pre")
    datasets: Optional[List[str]] = None
    cuts: Optional[Any] = None
    dummy_values = -99999 # value used to fill in missing values

    def __post_init__(self):
        # a dictionary of all files corresponding to a certain dataset
        self.datasets = find_datasets(self.dataset_pattern, self.eras, file_type="root", verbose=False)
    # TODO HASH ?



@dataclass
class ModelBuildingConfig:
    # Rotation Layer
    enable_rotation: bool = False # enable rotation layer based on two reference objects, TODO: this breaks export to torch script
    ref_phi_columns: Tuple[str, str] = ("res_dnn_pnet_vis_tau1", "res_dnn_pnet_vis_tau2") # reference column to calculate rotation angle
    rotate_columns: Tuple[str, ...] = (
        "res_dnn_pnet_bjet1",
        "res_dnn_pnet_bjet2",
        "res_dnn_pnet_fatjet",
        "res_dnn_pnet_vis_tau1",
        "res_dnn_pnet_vis_tau2",
    ) # which columns should be rotated

    # Padding Layers
    categorical_target_value: Optional[float] = None # which categorical value is targeted by the padding, if None no value is masked
    categorical_masking_value: Optional[float] = -1 # value that is used to mask the categorical target value

    continuous_target_value: Optional[float] = None # which continuous value is targeted by the padding, if None no value is masked
    continuous_masking_value: Optional[float] = EMPTY_FLOAT # value that is used to mask the continuous target value

    # Tokenizer & Embedding Layer
    tokenizer_add_unknown_category = None # value added to categories to symbolize unknown category. If None, no extra category is added
    embedding_dim: int = 10 # dimension of embedding layer - Marcel 10
    expected_embedding_inputs: Optional[dict[Tuple[int]]] = field(init=False)

    # DenseNet/LBN config
    nodes: int = 128 # base number of nodes of the dense blocks, due to DenseNet connection, increases with more layers
    activation_functions: str = "elu" # string of the activation function of DenseBlocks - Marcel DNN: elu
    skip_connection_init: float = 1 # init value of the skip connection, 1 = exact copy
    freeze_skip_connection: bool = True # True = non-learnable skip connection value
    batch_norm_eps: float = 0.001 # epsilon denominator of batch norm - increase stability Marcel: 0.001
    LBN_M: int = 10 # number of particles of the lbn network Marcel: 10
    last_activation_fn: LAST_ACTIVATION_CHOICE = "Softmax" # add activation function after last layer
    use_last_activation: bool = False # whether to use the last activation function, can be deactivated if not wanted - for example when using a loss function that already includes an activation like cross entropy, Marcel: False

    #STD Layers
    mean: Optional[Any] = None # these values are determined by your data
    std: Optional[Any] = None


    eps_batchnorm: int = 0.001 # epsilon of batch norm layers
    normalize_linear: bool = False # activate weight normalization of linear layer, TODO currently BUGGED, leave at False

    def __post_init__(self):
        self.expected_embedding_inputs = features.expected_embedding_inputs()
        choice_check(self.last_activation_fn, LAST_ACTIVATION_CHOICE)

        # currently parametrization and l2 maybe buggy
        if self.normalize_linear is True:
            raise ValueError("Norm of Linear Layer is currently buggy and is therefore disabled for now")



@dataclass
class BinningConfig:
    num_bins: int = 15
    bounds: tuple[int] = (0, 1)
    binning_fn: Any = torch.linspace  # keep as a callable, if needed
    kernel_cls: KERNEL_CHOICE = "GaussianKernelFinal"
    kernel_config: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "GaussianKernelFinal": {
                "abs_mode": False, # all values are interpreted as absolute values, recommended to be relative
                "smoothing_width": 0.1, # width of gaussian where it goes from 100% to 10%
                "left_notch": 0.0, # shift of gaussian into linear part from left
                "right_notch": 0.0, # shift of gaussian into linear part from right
                "bin_height" : 1,
            }
        }
    )
    def __post_init__(self):
        choice_check(self.kernel_cls, KERNEL_CHOICE)

@dataclass
class TrainingConfig:
    log_metrics: bool = True # whether to log metrics to tensorboard during training, if false only validation loss is logged
    model_choice: MODEL_CHOICE = "lbn_dense"
    loss_fn: LOSS_CHOICE = "cross_entropy"
    training_fn: TRAINING_LOOP_CHOICE = "cross_entropy" # name of the training loop
    validation_fn: VALIDATION_LOOP_CHOICE = "cross_entropy" # name of the validation loop
    loss_mode: SIGNAL_EFFICIENCY_LOSS_MODE = "no_unc" # only relevant if signal_efficiency loss is chosen, determines which formula is used for the asimov calculation
    loss_uncertainty: float = 0.0 # # only relevant if signal_efficiency loss is chosen, determines the background uncertainty used in the asimov calculation

    max_train_iteration: int = 60000 # max number of batches
    verbose_interval: int = 5 # interval between two logger outputs of training loss
    validation_interval: int = 100 # interval between two validation passes / plots are done during validation
    gamma: float = 0.5
    label_smoothing: float = 0.0
    train_folds: Tuple[int, ...] = (0,) # which training folds to use
    k_fold: int = 5
    seed: int = 100 # set torch and numpy seed for reproducibility
    train_ratio: float = 0.75 # split ratio for k-fold data into train and validation
    t_batch_size: int = 4096 * 10
    v_batch_size: int = -1 # validation batch size, -1 = full set,
    save_model_name: str = "dl_4-sl_1-fh_1" # name of the model used to save

    # Sampler Settings
    sample_ratio: Dict[str, float] = field(default_factory=lambda:{"dy": 1 / 3, "tt": 1 / 3, "hh": 1 / 3}) # decide the ratio of tt, dy and hh within a batch
    sub_process_ratios: Dict[str, float] = field(default_factory=lambda:{ # decide the
        # HINT: each rate is multiplied together to final rate, e.g. if process id exist 2x with rate 2, final rate is 4
        "signal":{}, # empty categorizes are set to 1 by default
        "tt_fh":{1300:1}, # groups are possible, and mixes are allowed
        "tt_sl": {1100:1},
        "tt_dl" : {1200:4},
        "dy":{51667: 1, 51683: 1, 51664: 1, 51680: 1, 51720: 1, 51723: 1, 51726: 1,
        51729: 1, 51732: 1, 51735: 1, 51674: 1, 51690: 1, 51665: 1, 51681: 1,
        51661: 1, 51677: 1, 51670: 1, 51671: 1, 51672: 1, 51673: 1, 51675: 1,
        51686: 1, 51687: 1, 51688: 1, 51689: 1, 51691: 1, 51666: 1, 51682: 1,
        51699: 1, 51702: 1, 51705: 1, 51708: 1, 51711: 1, 51714: 1, 51693: 1,
        51668: 1, 51684: 1, 51663: 1, 51679: 1,
        },
    })
    use_sub_process_ratios: Tuple[str] = ("signal", "tt_fh", "tt_sl", "tt_dl", "dy")
    sample_attributes: Tuple[str, ...] = (
        "continuous",
        "categorical",
        "targets",
        "product_of_weights",
        "evaluation_space_mask",
    ) # what to sample from sampler
    min_events_in_batch: int = 1 # sampler setting: minimal number of events of a subprocess in a batch

    def __post_init__(self):
        necessary_field = ("continuous", "categorical", "targets")
        if not set(necessary_field).issubset(set(self.sample_attributes)):
            raise ValueError(f"Sample Attributes need to contain: {necessary_field}" )
        choice_check(self.training_fn, TRAINING_LOOP_CHOICE)
        choice_check(self.validation_fn, VALIDATION_LOOP_CHOICE)
        choice_check(self.loss_mode, SIGNAL_EFFICIENCY_LOSS_MODE)
        choice_check(self.loss_fn, LOSS_CHOICE)
        choice_check(self.model_choice, MODEL_CHOICE)
        self.sub_process_ratios = multiply_sub_process_rates(self.use_sub_process_ratios, self.sub_process_ratios)



@dataclass
class SchedulerConfig:
    scheduler_chain: Tuple[SCHEDULER_CHOICE, ...] = ("linear", "cosine_annealing") # schedulers used in chain
    config_chain: Optional[Tuple[Any, ...]] = None # list of configs corresponding to the schedulers in the scheduler chain
    scheduler_cls_chain: Optional[Tuple[Any, ...]] = None # list of scheduler classes corresponding to the schedulers in the scheduler chain, if None is given, it is assumed that the scheduler class can be derived from the scheduler choice by adding "LR" at the end, for example "CosineAnnealingLR" for "cosine"
    milestones: Tuple[int, ...] = (500,) # intervals after which the LR scheduler is swapped


    @dataclass
    class StepLRConfig(): # used by marcel
        step_size: int = 10 # number of iterations between two learning rate reductions
        gamma: float = 0.5 # learning rate reduction factor
        # min_delta: float = 0.0 # minimum improvement before increase patience - Marcel: 0


    @dataclass
    class CosineAnnealingLRConfig():
        T_max: int = 15000 # maximum number of iterations
        eta_min: float = 1e-7 # minimum learning rate

    @dataclass
    class ReduceLROnPlateauConfig():
        mode: str ='min' # one of {min, max}, in min mode lr will be reduced when the quantity monitored has stopped decreasing
        patience: int = 4 # wait x number of checks - Marcel: 10
        threshold_mode: str = "abs" # type of min_delta - Marcel: abs
        factor: float = 0.5 # LR reduce by factor
        cooldown: int = 0, # number of iterations to wait after a learning rate reduction before resuming normal operation
        min_lr: float = 0, # lower bound on the learning rate
        eps: float = 1e-08,  # minimal decay applied to lr, if it is smaller than this value, it is set to this value

    @dataclass
    class LinearLRConfig():
        start_factor: float = 0.01 # the initial learning rate will be the start_factor times the base learning rate
        end_factor: float = 1.0 # the final learning rate will be the end_factor times the base learning rate
        total_iters: int = 100 # number of iterations over which the multiplier increases from start_factor to end_factor


    def __post_init__(self):
        schedulers = torch.optim.lr_scheduler

        scheduler_register = {
            "cosine_annealing" : (self.CosineAnnealingLRConfig, schedulers.CosineAnnealingLR),
            "reduce_on_plateau" : (self.ReduceLROnPlateauConfig, schedulers.ReduceLROnPlateau),
            "linear" : (self.LinearLRConfig, schedulers.LinearLR),
            "step" : (self.StepLRConfig, schedulers.StepLR),
        }

        scheduler_configs = []
        scheduler_cls_chain = []
        for choice in self.scheduler_chain:
            config, scheduler_cls = scheduler_register[choice]
            scheduler_configs.append(asdict(config()))
            scheduler_cls_chain.append(scheduler_cls) # init is done in the training
            choice_check(choice, SCHEDULER_CHOICE)
        self.config_chain = tuple(scheduler_configs)
        self.scheduler_cls_chain = tuple(scheduler_cls_chain)

@dataclass
class OptimizerConfig:
    optimizer_choice: OPTIMIZER_CHOICE = "adamw"

    @dataclass
    class ADAMWConfig():
        decay_factor: float = 500 # factor of L2 - Marcel: 500
        normalize: bool = True # normalize weight decay factor to number of parameters
        lr: float = 1e-3 # start learning rate

    @dataclass
    class SAMConfig():
        TODO = None

    def __post_init__(self):
        # TODO change like in scheduler
        optimizer_register = {
            "adamw" : self.ADAMWConfig,
            "sam" : self.SAMConfig,
        }

        # move optimizer attribute to top level
        choice_config = optimizer_register[self.optimizer_choice]()

        for _key, _attr in choice_config.__dict__.items():
            setattr(self, _key, _attr)

        choice_check(self.optimizer_choice, OPTIMIZER_CHOICE)


@dataclass
class DebugConfig:
    get_batch_statistic_return_dummy: bool = False # do not calculate preprocessing, instead use mean = 0, std = 1
    load_marcel_stats: bool = False # load Marcels mean and std from preprocessing into STD Layer
    load_marcel_weights: bool = False # preload Marcels Pytorch Model weights, HINT: will break now, since architecture changes


# Combination of all Configs so only 1 import is necessary
@dataclass
class FullConfig:
    dataset_config: DataConfig = field(default_factory=DataConfig)
    model_building_config: ModelBuildingConfig = field(default_factory=ModelBuildingConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    binning_config: BinningConfig = field(default_factory=BinningConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    debug_config: DebugConfig = field(default_factory=DebugConfig)

    def __post_init__(self):
        # when using optimizer sam specific training routine needs to be used
        if (self.training_config.training_fn != "sam") & (self.optimizer_config.optimizer_choice=="sam"):
            raise ValueError(f"When using Optimizer SAM, training_fn needs to be set to 'sam', is currently:{self.training_config.training_fn}).")


full_config = FullConfig()
