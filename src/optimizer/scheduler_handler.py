import torch


class SchedulerHandler():
    def __init__(self, scheduler_inst, checkpoint_inst, logger_inst):
        self.scheduler_inst = scheduler_inst
        self.checkpoint_inst = checkpoint_inst
        self.logger_inst = logger_inst
        self._needs_metric = isinstance(
            scheduler_inst,
            (torch.optim.lr_scheduler.ReduceLROnPlateau,)
            )


    def step(
        self,
        model_inst: torch.nn.Module,
        optimizer_inst : torch.optim.Optimizer,
        metric: torch.Tensor=None
        ) -> bool:
        """
        Call once per evaluation.
        There are two place where a step happens: Once per batch OR when a metric is evaluated.
        The latter is at the end of the training, while the former is direct after the optimizer step.
        The correct one is chosen depending on _needs_metric and chosen scheduler_inst.

        Args:
            model_inst (torch.nn.Module): Passed ML model instance.
            optimizer_inst (torch.optim.Optimizer): Passed Optimizer instance.
            metric (torch.Tensor, optional): _description_. Defaults to None.
        """

        if self.scheduler_inst is None:
            return False

        # check mutual exclusivity:
        # when a metric is given but scheduler does not need it
        # OR when a metric that needs a metric has none
        # do nothing in these cases
        metric_given = metric is not None
        if metric_given != self._needs_metric:
            return False


        # step for plateau scheduler is different
        previous_lr = optimizer_inst.param_groups[0]["lr"]
        if self._needs_metric:
            self.scheduler_inst.step(metric)
        else:
            self.scheduler_inst.step()
        current_lr = optimizer_inst.param_groups[0]["lr"]

        if previous_lr == current_lr:
            return False

        return self._reload_after_lr_drop(
            model_inst=model_inst,
            optimizer_inst=optimizer_inst,
            previous_lr=previous_lr,
            current_lr=current_lr,
            )


    def _reload_after_lr_drop(
        self,
        model_inst: torch.nn.Module,
        optimizer_inst: torch.optim.Optimizer,
        previous_lr: torch.Tensor,
        current_lr: torch.Tensor
        ) -> bool:
        checkpoint = self.checkpoint_inst.last_checkpoint

        if checkpoint is None:
            return False

        self.logger_inst.info(
            f"{previous_lr} -> {current_lr}" +
            "\nReload model and optimizer from iteration"
            f" {checkpoint['iteration']}")

        model_inst.load_state_dict(checkpoint["model_state_dict"])
        optimizer_inst.load_state_dict(checkpoint["optimizer_state_dict"])
        # since scheduler and optimizer are coupled
        # overwrite old lr in checkpoint with lr after scheduler step
        for group in optimizer_inst.param_groups:
            group["lr"] = current_lr

        return True