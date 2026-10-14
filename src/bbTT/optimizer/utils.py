import torch

from bbTT.optimizer import weight_decay


def init_optimizer(full_config, model_inst):
    # only linear layers contribute to weight decay, prepare config that separates them for the optimizer
    weight_decay_parameters = weight_decay.normalized_weight_decay(
        model_inst,
        full_config.optimizer_config.active_config.decay_factor,
        full_config.optimizer_config.active_config.normalize,
    )
    # is a dictionary, expect list of tuples
    weight_decay_parameters = list(weight_decay_parameters.values())
    name = full_config.optimizer_config.optimizer_choice

    global_weight_decay_default = 0

    # pick correct optimizer
    if name == "adamw":
        optimizer_inst = torch.optim.AdamW(
            weight_decay_parameters,
            lr=full_config.optimizer_config.active_config.lr,
            weight_decay=global_weight_decay_default,
        )
    else:
        raise ValueError(f"Chosen Optimizer {name} does not exist")
    return optimizer_inst


def init_scheduler(full_config, optimizer_inst):
    s_cfg = full_config.scheduler_config

    scheduler_instances = []
    for scheduler_cfg, scheduler_name in zip(s_cfg.config_chain, s_cfg.scheduler_cls_chain):
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_name)
        scheduler_instances.append(scheduler_cls(optimizer_inst, **scheduler_cfg))

    # sequential LR does not enable adaptive scheduler
    if len(scheduler_instances) == 1:
        scheduler_inst = scheduler_instances[0]
    else:
        scheduler_inst = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer_inst, schedulers=scheduler_instances, milestones=s_cfg.milestones
        )
    return scheduler_inst
