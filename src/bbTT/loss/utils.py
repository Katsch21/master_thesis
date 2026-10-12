
from bbTT.loss import SignalEfficiency, WeightedCrossEntropy
from bbTT.utils.utils import CPU_DEVICE


def init_loss(full_config, device=CPU_DEVICE, **kwargs):
    cfg = full_config.loss_config

    if cfg.loss_fn == "cross_entropy":
        # the default trainings_loss_function
        train_loss_fn = WeightedCrossEntropy(
            weight=None, size_average=None, label_smoothing=full_config.training_config.label_smoothing
        )
        validation_loss_fn = WeightedCrossEntropy(
            weight=None, size_average=None, label_smoothing=full_config.training_config.label_smoothing
        )

    elif cfg.loss_fn == "signal_efficiency":
        loss_cfg = {
            "sampler_inst": kwargs["training_sampler"],
            "device": device,
            "asimov_cfg": cfg.active_config,
        }

        train_loss_fn = SignalEfficiency(is_training=True, **loss_cfg)
        validation_loss_fn = SignalEfficiency(is_training=False, **loss_cfg)
    return train_loss_fn, validation_loss_fn
