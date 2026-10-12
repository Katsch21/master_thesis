from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from bbTT.monitoring.logger.logger import get_logger

if TYPE_CHECKING:
    from data_handling.sampler import ProcessSampler

logger_inst = get_logger(__name__)


def register_loop(name):
    # add markers to function for later registration in loop registry
    def decorator(fn):
        fn._is_loop_method = True
        fn._loop_name = name
        return fn

    return decorator


class BaseLoop:
    """
    Base Class that implements are registry where registered loops are categorized
    as training and validation loops.
    The registry is used to check if a given loop function exists and to call the correct function in the training script.
    """

    REGISTERED_LOOPS = {}
    MODE = None  # is overwritten by child class

    def __init__(
        self,
        full_config,
        necessary_columns: set[str] | None = None,
        *args,
        **kwargs):
        self.full_config = full_config
        self.which_fn = (
            self.full_config.training_config.training_fn
            if self.MODE == "training"
            else self.full_config.training_config.validation_fn
        )
        # these column are always necessary for the loops
        self.necessary_columns = (
            {"continuous", "categorical", "targets"} if necessary_columns is None else necessary_columns
        )
        self.target_shape = (-1, len(self.full_config.dataset_config.target_map))
        self.target_map = self.full_config.dataset_config.target_map

    @property
    def batch_size(self):
        if self.MODE == "training":
            return self.full_config.training_config.training_batch_size
        else:
            return self.full_config.training_config.validation_batch_size

    def __init_subclass__(cls):
        for _name, fn in cls.__dict__.items():
            # only add methods if is part of self
            if getattr(fn, "_is_loop_method", False):
                cls.REGISTERED_LOOPS.setdefault(cls.MODE, {})  # make sure that dict for mode exists
                cls.REGISTERED_LOOPS[cls.MODE][fn._loop_name] = fn

    def separate_prediction(
        self,
        pred: torch.tensor | tuple[torch.tensor],
    ) -> tuple[torch.tensor, torch.tensor]:
        """
        Depending on the network architecture the output can either be a tuple of prediction and binned prediction
        or just the prediction. To have a uniform interface the singular output uses a dummy value.

        Args:
            pred (torch.tensor | tuple[torch.tensor]): _description_

        Returns:
            tuple[torch.tensor, torch.tensor]: Tuple of predictions where first value is used for logging and second value is used for optimization.
            In case of non binned output both values are the same.
        """
        # TODO maybe replace with dictionary?
        is_binned = isinstance(pred, tuple)
        if is_binned:
            logging_pred, optimization_pred = pred
        else:
            logging_pred, optimization_pred = pred, pred

        return logging_pred, optimization_pred

    def setup(self, monitor=None, model_inst=None, kind_of_data="batch"):
        if monitor is not None:
            monitor.set_mode(kind_of_data)
            monitor.start_step(kind_of_data)

    def cleanup(self, monitor, model_inst, **kwargs):
        pass

    def __call__(
        self,
        *args,
        monitor=None,
        model_inst=None,
        kind_of_data=None,
        scheduler_handler_inst=None,
        optimizer_inst=None,
        **kwargs,
    ):
        # registered loop is just an unbound function
        # bound by passing self as first argument
        self.setup(monitor=monitor, model_inst=model_inst, kind_of_data=kind_of_data)
        try:
            fn = self.REGISTERED_LOOPS[self.MODE][self.which_fn]
            result = fn(self, *args, monitor=monitor, model_inst=model_inst, optimizer_inst=optimizer_inst, **kwargs)
        finally:
            self.cleanup(
                monitor=monitor,
                model_inst=model_inst,
                scheduler_handler_inst=scheduler_handler_inst,
                optimizer_inst=optimizer_inst,
            )
        return result


class TrainingLoop(BaseLoop):
    MODE = "training"

    def __init__(self, full_config, *args, **kwargs):
        super().__init__(*args, **kwargs, full_config=full_config)

    def setup(self, monitor=None, model_inst=None, kind_of_data="batch"):
        super().setup(monitor, model_inst, kind_of_data)
        model_inst.enable_gradient_hooks()

    def cleanup(self, monitor=None, model_inst=None, optimizer_inst=None, scheduler_handler_inst=None, **kwargs):
        super().cleanup(monitor=monitor, model_inst=model_inst)
        if scheduler_handler_inst is not None:
            scheduler_handler_inst.step(model_inst=model_inst, optimizer_inst=optimizer_inst, metric=None)

        model_inst.disable_gradient_hooks()
        if monitor is not None:
            monitor.check_gradient_correctness(expected_names=model_inst.monitored_gradient_names())

    @register_loop(name="sam")
    def sam_optimizer(
        self,
        model_inst,
        loss_fn,
        optimizer_inst,
        sampler,
        device,
        *args,
        **kwargs,
    ):
        optimizer_inst.zero_grad()

        events = sampler.sample_batch(device=device)
        categorical, continuous = events.get("categorical"), events.get("continuous")
        targets = events["targets"].reshape(self.target_shape)

        pred = model_inst(categorical_inputs=categorical, continuous_inputs=continuous)
        predictions, optimization_predictions = self.separate_prediction(pred)

        loss = loss_fn(
            optimization_predictions,
        )
        loss.backward()

        # optimizer routine differs for sam optimizers
        # first step included scouting, second step is the actual optimized step, batch norm needs to be disabled for that
        optimizer_inst.first_step(zero_grad=True)

        # second forward step with disabled batchnorm running stats in second forward step
        optimizer_inst.disable_running_stats(model_inst)
        pred_2 = model_inst(categorical_inputs=categorical, continuous_inputs=continuous)
        loss_fn(pred_2, targets).backward()

        optimizer_inst.second_step(zero_grad=True)
        optimizer_inst.enable_running_stats(model_inst)  # <- this is the important line

        return {
            "loss": loss,
            "predictions": predictions,
            "targets": targets,
            "event_weights": events["product_of_weights"],
        }

    @register_loop(name="cross_entropy")
    def cross_entropy_loss(
        self,
        model_inst,
        loss_fn,
        optimizer_inst,
        sampler,
        device,
        sample_columns=None,
        monitor=None,
        **kwargs,
    ):
        # this loss never uses a binning network thus, a single prediction is expected
        optimizer_inst.zero_grad()

        events = sampler.sample_batch(sample_from=sample_columns, device=device)
        categorical, continuous = events.get("categorical"), events.get("continuous")
        targets = events["targets"].reshape(self.target_shape)
        predictions = model_inst(categorical_inputs=categorical, continuous_inputs=continuous)

        # training does not need event weights. The oversampling algorithm takes care of this
        loss = loss_fn(predictions, targets, event_weights=None)
        loss.backward()
        optimizer_inst.step()
        return {
            "loss": loss,
            "predictions": predictions,
            "targets": targets,
            "event_weights": events["product_of_weights"],
        }

    @register_loop(name="signal_efficiency")
    def signal_efficiency_loop(
        self,
        model_inst,
        loss_fn,
        optimizer_inst,
        sampler,
        sample_columns,
        device,
        **kwargs,
    ):
        optimizer_inst.zero_grad()

        events = sampler.sample_batch(sample_from=sample_columns, device=device)
        categorical, continuous = events.get("categorical"), events.get("continuous")
        targets = events["targets"].reshape(self.target_shape)

        pred = model_inst(
            categorical_inputs=categorical,
            continuous_inputs=continuous,
        )

        predictions, optimization_predictions = self.separate_prediction(pred)

        loss = loss_fn(
            prediction=optimization_predictions,
            truth=targets,
            product_of_weights=events["product_of_weights"],
            evaluation_mask=events["evaluation_space_mask"],
        )

        loss.backward()
        optimizer_inst.step()
        return {
            "loss": loss,
            "predictions": predictions,
            "targets": targets,
            "event_weights": events["product_of_weights"],
        }


class ValidationLoop(BaseLoop):
    MODE = "validation"

    def __init__(self, full_config, *args, **kwargs):
        super().__init__(*args, **kwargs, full_config=full_config)

    def setup(self, monitor=None, model_inst=None, kind_of_data="batch"):
        super().setup(monitor, model_inst, kind_of_data)
        # evaluation never look into gradients
        model_inst.disable_gradient_hooks()

    def cleanup(self, monitor, model_inst, **kwargs):
        super().cleanup(monitor=monitor, model_inst=model_inst)

    def collect_from_generators(
        self,
        model_inst: torch.nn.Module,
        sampler_inst: ProcessSampler,
        sample_columns: list[str],
        skip_concatenate_columns: tuple[str] = ("",),
        device: str | torch.device = "cpu",
        *args,
        **kwargs,
    ) -> dict[torch.tensors]:
        """
        During Validation no sampling is used, instead we loop over all unsampeled Datasets.
        The *model_inst* is the DNN you want to use, while *sampler_inst* should be a ValidationSampler.
        Which column is aggregated over all datasets is defined in *sample_columns*.
        In the end a concatenation is done over the event axis. If this step should be skipped for
        certain columns (e.g. to maintain the shape), add the name in *skip_concatenate_columns*.

        Args:
            model_inst (torch.nn.Module): Your neural network model.
            sampler_inst (ProcessSampler): Instance of a ProcessSampler, with registered Processes.
            sample_columns (list[str]): List of columns to sample from.
            skip_concatenate_columns (tuple[str], optional): List of columns to skip concatenation. Defaults to ("",).
            device (torch.device, optional): Device everything should run on. Defaults to torch.device("cpu").

        Returns:
            dict[torch.tensors]: Dictonary with all aggregated tensors.
        """
        device = torch.device(device) if isinstance(device, str) else device

        model_inst.eval()
        collected_data = {
            "class_predictions": [],
            "targets": [],
            "sample_weights": [],
            "loss_predictions": [],
            "relative_weights": [],
            "post_relative_weights": [],
            "process_id": [],
        }

        dynamic_data = {dynamic_d: [] for dynamic_d in tuple(sample_columns - self.necessary_columns)}

        with torch.no_grad():
            # sample over all dataset generator and run network
            # store output in lists which are then concatenated in the end
            for uid, events in sampler_inst.full_pass(
                batch_size=self.batch_size, sample_from=sample_columns, device=device
            ):
                # hold data for current dataset, saved in loop to avoid memory issues
                current_process = sampler_inst.registry[uid]
                pred = model_inst(
                    categorical_inputs=events.pop("categorical"), continuous_inputs=events.pop("continuous")
                )
                # network can output either binned or non binned predictions
                # depending on model architecture, separate_prediction gives us a uniform interface to handle both cases
                class_pred, loss_pred = self.separate_prediction(pred)

                # -- collect data that is always done --

                collected_data["targets"].append(events.pop("targets"))
                collected_data["sample_weights"].append(events.pop("sample_weights"))

                collected_data["relative_weights"].append(
                    torch.full(size=(class_pred.shape[0], 1), fill_value=current_process.relative_weight, device=device)
                )
                collected_data["post_relative_weights"].append(
                    torch.full(size=(class_pred.shape[0], 1), fill_value=current_process.post_relative_weight, device=device)
                )

                collected_data["process_id"].append(
                    torch.full((class_pred.shape[0], 1), uid[1], device=device)
                    )

                if not model_inst.use_last_activation:
                    # TODO if sigmoid is necessary add switch case
                    class_pred = torch.softmax(class_pred, dim=1)
                collected_data["class_predictions"].append(class_pred)

                # for loss calculation, can be the same as class_prediction e.g. crossentropy
                collected_data["loss_predictions"].append(loss_pred)

                # -- collect dynamic data --
                for k in dynamic_data.keys():
                    dynamic_data[k].append(events.pop(k))

                # it is necessary to ensure the same shape
                # when mapping of activation function is on (only having softmax) do not apply them, else do apply

            # combine datas
            collected_data.update(dynamic_data)

            for k in tuple(collected_data.keys()):
                value = collected_data.pop(k)
                # event dimension is always the second last axis
                event_dim = len(value[0].shape) - 2
                # for whatever reason maybe want to skip concatenate?
                if k not in skip_concatenate_columns:
                    value = torch.concatenate(value, dim=event_dim)
                collected_data[k] = value
        return collected_data

    @register_loop(name="cross_entropy")
    def cross_entropy_loop(
        self,
        model_inst,
        loss_fn_inst,
        sampler_inst,
        sample_columns,
        device,
        monitor=None,
        *args,
        **kwargs,
    ):
        """
        Used for normal loss functions like Crossentropy.
        """
        to_sample_columns = self.necessary_columns.union(sample_columns)
        tensors = self.collect_from_generators(
            model_inst=model_inst,
            sampler_inst=sampler_inst,
            sample_columns=to_sample_columns,
            skip_concatenate_columns=("",),
            device=device,
        )
        # validation needs to be reweightes by sample weights
        loss = loss_fn_inst(
            tensors["loss_predictions"],
            tensors["targets"],
            tensors["post_relative_weights"],
        )
        return {
            "loss": loss,
            "predictions": tensors["class_predictions"],
            "targets": tensors["targets"],
            "event_weights": tensors["product_of_weights"],
        }

    @register_loop(name="signal_efficiency")
    def signal_efficiency_loop(
        self,
        model_inst,
        loss_fn_inst,
        sampler_inst,
        sample_columns,
        device,
        monitor=None,
        *args,
        **kwargs,
    ):
        loss_columns = {"product_of_weights", "evaluation_space_mask"}
        to_sample_columns = self.necessary_columns.union(loss_columns).union(sample_columns)

        tensors = self.collect_from_generators(
            model_inst=model_inst,
            sampler_inst=sampler_inst,
            sample_columns=to_sample_columns,
            skip_concatenate_columns=("",),
            device=device,
        )

        loss = loss_fn_inst(
            tensors["loss_predictions"],
            tensors["targets"].reshape(self.target_shape),
            tensors["product_of_weights"],
            tensors["evaluation_space_mask"],
            # TODO  sampler weight is not used, but maybe should ?
        )
        return {
            "loss": loss,
            "predictions": tensors["class_predictions"],
            "targets": tensors["targets"],
            "event_weights": tensors["sample_weights"],
        }
