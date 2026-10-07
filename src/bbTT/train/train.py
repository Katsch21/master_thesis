from __future__ import annotations

# standard imports
import dataclasses

# package imports
import numpy as np
import torch

import bbTT.data_handling.sampling.sampler as sampler
from bbTT.configs.full_config import FullConfig

# personal imports
from bbTT.data_handling.io import get_data
from bbTT.data_handling.preprocessing.k_fold import FoldAndSplitCoordinator
from bbTT.data_handling.preprocessing.standardization import FeatureStatisticCache
from bbTT.data_handling.sampling.weight import WeightAggregator
from bbTT.data_handling.utils import hash_dictionary
from bbTT.loss import init_loss
from bbTT.models.utils import init_model

# from .train_utils import log_metrics
from bbTT.monitoring import (
    EvalContext,
    EvaluationRunner,
    TrainingMonitor,
    load_registers,
    setup_monitoring,
)
from bbTT.monitoring.context.batch_composition_context import BatchCompositionHistory
from bbTT.monitoring.logger.logger import get_logger
from bbTT.monitoring.logger.tensorboard_logger import TensorboardLogger
from bbTT.optimizer.early_stopping import CheckPoint
from bbTT.optimizer.scheduler_handler import SchedulerHandler
from bbTT.optimizer.utils import init_optimizer, init_scheduler
from bbTT.train.loops import TrainingLoop, ValidationLoop
from bbTT.utils.utils import DEVICE

from collections import deque

full_config = FullConfig()
torch.manual_seed(full_config.training_config.seed)
np.random.seed(full_config.training_config.seed)


def main(**kwargs):
    # prepare logger
    logger_inst = get_logger(__name__)
    logger_inst.info(f"DEVICE: {DEVICE}")
    tensorboard_writer = TensorboardLogger(
        name=hash_dictionary(dataclasses.asdict(full_config.training_config)),
        path=kwargs["tensorboard_name"],
    )
    evaluation_runner_inst = EvaluationRunner(tensorboard_writer)
    # load all registered plots and metrics
    load_registers()
    # TODO add LOGGER FILEPATH in same directory of tensorboard
    logger_inst.i_info(f"Tensorboard logs: {tensorboard_writer.path}")

    # load data
    for current_fold in full_config.training_config.train_folds:
        logger_inst.info(
            f"Trainings fold: {current_fold}/{full_config.training_config.k_fold - 1}"
        )
        # -----
        ### data loading and preprocessing
        # -----
        # events is of form : {uid : {"COLUMN_NAME": torch.Tensor}}
        events = get_data(
            full_config.dataset_config,
            ignore_cache=kwargs["ignore_cache"],
            save_cache=kwargs["save_cache"],
        )

        # split data into training, validation or test set - according to fold
        # HINT: order matters, due to memory constraints, events is changed inplace.
        #  To get Weight statistics all events need to be present. Thus this needs to run before apply splitting

        # Calculate all Indices for Splitting
        fold_split_coordinator = FoldAndSplitCoordinator(
            events=events,
            c_fold=current_fold,
            k_fold=full_config.training_config.k_fold,
            seed=full_config.training_config.seed,
            training_percentage=0.75,
            randomize=True,
        )

        columns_to_split = (
            "continuous",
            "categorical",
            "event_id",
            "normalization_weights",
            "product_of_weights",
            "evaluation_mask",
        )

        # calculate weight statistics, NEEDS TO RUN BEFORE APPLY INDICES
        weight_aggregator = WeightAggregator(events, fold_split_coordinator.indices)
        # actually apply indices, but release on the fly to reduce peak memory
        split_events = fold_split_coordinator.split_and_free(events, kinds=("training", "validation"), columns=columns_to_split)
        train_events, validation_events = split_events["training"], split_events["validation"]
        del events

        # create new sampler, splitted dict are released
        logger_inst.info("Create Sampler")
        training_sampler = sampler.create_sampler(
            train_events,
            train=True,
            weight_aggregator_inst=weight_aggregator,
            full_config=full_config,
        )
        del train_events

        validation_sampler = sampler.create_sampler(
            validation_events,
            train=False,
            weight_aggregator_inst=weight_aggregator,
            full_config=full_config,
        )
        del validation_events

        # load calculated sampler from training sampler to validation sampler
        validation_sampler.load_process_weights_from(training_sampler)

        # calculate or load statistic of DATA used to for training, this information is necessary for standardization.
        feature_statistic_cache = FeatureStatisticCache(
            dataset_config=full_config.dataset_config,
            sampler=training_sampler,
            return_dummy=full_config.debug_config.get_batch_statistic_return_dummy,
            verbose=True,
        )
        full_config.model_building_config.standardization.mean = feature_statistic_cache.mean
        full_config.model_building_config.standardization.std = feature_statistic_cache.std

        # ----
        ### model build and configuration, including optimizer, scheduler, early stopping and loss function
        # ----
        model_inst = init_model(full_config=full_config)
        model_inst = model_inst.to(DEVICE).train()
        training_loop, validation_loop = (
            TrainingLoop(full_config),
            ValidationLoop(full_config),
        )

        optimizer_inst = init_optimizer(full_config=full_config, model_inst=model_inst)
        training_loss_inst, validation_loss_inst = init_loss(
            full_config=full_config, device=DEVICE, training_sampler=training_sampler
        )
        scheduler_inst = init_scheduler(
            full_config=full_config, optimizer_inst=optimizer_inst
        )
        checkpoint_inst = CheckPoint(
            checkpoint_name=full_config.training_config.save_model_name,
            checkpoint_fold=current_fold,
        )
        training_monitor_inst = TrainingMonitor(to_cpu=True, non_blocking=True)
        scheduler_handler_inst = SchedulerHandler(
            scheduler_inst=scheduler_inst,
            checkpoint_inst=checkpoint_inst,
            logger_inst=logger_inst,
        )
        mode_batch, mode_eval_training, mode_eval_validation = (
            "training_batch",
            "evaluation_training",
            "evaluation_validation",
        )
        batch_composition_history_inst = BatchCompositionHistory()

        setup_monitoring(
            training_monitor_inst,
            model_inst,
            # model_inst.binning_layer,
            training_loss_inst,
            validation_loss_inst,
        )

        # ----
        ### training loop
        # ----
        logger_inst.info("Start training loop")
        last_losses = deque([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], maxlen=10)
        for current_iteration in range(full_config.training_config.max_train_iteration):
            batch_result = training_loop(
                model_inst=model_inst,
                monitor=training_monitor_inst,
                kind_of_data=mode_batch,
                loss_fn=training_loss_inst,
                sampler=training_sampler,
                device=DEVICE,
                sample_columns=full_config.sampler_config.sample_attributes,
                scheduler_handler_inst=scheduler_handler_inst,
                optimizer_inst=optimizer_inst,
            )

            if full_config.record_config.batch_metrics:
                batch_composition_history_inst.record(
                    step=current_iteration, cursors=training_sampler.cursors
                )

            # ----
            # Verbose and Metrics that are triggered often
            # ----
            if current_iteration % full_config.record_config.verbose_interval == 0:
                tensorboard_writer.log_lr(optimizer_inst, current_iteration)
                batch_loss = batch_result["loss"].item()
                tensorboard_writer.log_loss(
                    {"batch_loss": batch_loss}, step=current_iteration
                )
                current_lr = optimizer_inst.param_groups[0]["lr"]
                logger_inst.training(
                    f"T-It: {current_iteration} - LR: {current_lr} - batch loss: {batch_loss:.2E}"
                )

            # ----
            #### Evaluation of training and validation data, logging and checkpointing
            # ----
            evaluation_condition = (
                current_iteration % full_config.record_config.validation_interval == 0
            ) & (current_iteration >= 0)
            if evaluation_condition:
                # evaluation of training data
                logger_inst.info(
                    f"Iteration {current_iteration}. Start evaluation of training data."
                )

                evaluation_training_result = validation_loop(
                    model_inst=model_inst,
                    monitor=training_monitor_inst,
                    kind_of_data=mode_eval_training,
                    loss_fn_inst=validation_loss_inst,
                    sampler_inst=training_sampler,
                    sample_columns=full_config.sampler_config.sample_attributes,
                    device=DEVICE,
                )
                # evaluation of validation
                logger_inst.info(
                    f"Iteration {current_iteration}. Start evaluation of validation data."
                )

                evaluation_validation_result = validation_loop(
                    model_inst=model_inst,
                    monitor=training_monitor_inst,
                    kind_of_data=mode_eval_validation,
                    loss_fn_inst=validation_loss_inst,
                    sampler_inst=validation_sampler,
                    sample_columns=full_config.sampler_config.sample_attributes,
                    device=DEVICE,
                )

                eval_t_loss = evaluation_training_result["loss"].item()
                eval_v_loss = evaluation_validation_result["loss"].item()
                logger_inst.training(
                    f"Iteration: {current_iteration} - TLoss: {eval_t_loss:.2E} VLoss: {eval_v_loss:.2E}"
                )

                # TODO when edges should be tracked add this in a way that is universal and does not break for models without binning layer, e.g. add property to model that returns None if no binning layer is present and add check in log_metrics
                if full_config.record_config.log_metrics:
                    # --- plots on evaluation, on training data ---
                    model_evaluation_state = model_inst.evaluation_state()

                    shared_eval_context_meta_data = {
                        "model_evaluation_state": model_evaluation_state,
                        "target_map": full_config.dataset_config.target_map,
                        "global_step": current_iteration,
                        "default_n_bins": full_config.binning_config.num_bins,
                        "batch_composition": batch_composition_history_inst,
                    }

                    ctx_batch = EvalContext(
                        mode="batch",
                        predictions=batch_result["predictions"],
                        targets=batch_result["targets"],
                        event_weights=batch_result["event_weights"],
                        **shared_eval_context_meta_data,
                    )

                    ctx_train = EvalContext(
                        mode="training",
                        predictions=evaluation_training_result["predictions"],
                        targets=evaluation_training_result["targets"],
                        event_weights=evaluation_training_result["event_weights"],
                        **shared_eval_context_meta_data,
                    )

                    ctx_validation = EvalContext(
                        mode="validation",
                        predictions=evaluation_validation_result["predictions"],
                        targets=evaluation_validation_result["targets"],
                        event_weights=evaluation_validation_result["event_weights"],
                        **shared_eval_context_meta_data,
                    )
                    # ctx_batch.add_feature("kernels", model_inst.kernels)
                    # ctx_batch.add_feature("binning_fn", model_inst.binning_fn)

                    ctx_batch.add_features(
                        *training_monitor_inst.get_plot_gradients(mode_batch),
                        *training_monitor_inst.get_plot_tensors(mode_batch),
                    )

                    ctx_train.add_features(
                        *training_monitor_inst.get_plot_tensors(mode_eval_training),
                        ("loss", eval_t_loss),
                    )

                    ctx_validation.add_features(
                        *training_monitor_inst.get_plot_tensors(mode_eval_validation),
                        ("loss", eval_v_loss),
                    )

                    # --- Plotting
                    evaluation_runner_inst.run_plots(
                        ctx_batch,
                        plots=[
                            # "kernels_monitor",
                            "batch_composition_history",
                        ],
                    )
                    # run metrics and store them
                    evaluation_runner_inst.run_plots(
                        ctx_train,
                        plots=[
                            "confusion_matrix",
                            # "roc", # roc computation is the long factor (14s)
                            "output_score_hh_node",
                            "output_score_hh_node_untransformed",
                            "kernels_monitor",
                            "score_correlation_matrix",
                            # "precision_recall", # deactivated takes 40s to use
                        ],
                    )

                    evaluation_runner_inst.run_plots(
                        ctx_validation,
                        plots=[
                            "confusion_matrix",
                            # "roc", # roc computation is the long factor (4s)
                            "output_score_hh_node",
                            "output_score_hh_node_untransformed",
                            "kernels_monitor",
                            "score_correlation_matrix",
                            # "precision_recall", # takes to long
                        ],
                    )

                    evaluation_runner_inst.run_scalars(
                        ctx=ctx_train,
                        artifact_names={
                            "CrossEntropy/Evaluation Training": "cross_entropy",
                            "Loss/Evaluation Training Loss": "loss",
                        },
                    )

                    evaluation_runner_inst.run_scalars(
                        ctx=ctx_validation,
                        artifact_names={
                            "CrossEntropy/Evaluation Validation": "cross_entropy",
                            "Loss/Validation VLoss": "loss",
                        },
                    )

                last_losses.append(eval_v_loss.cpu().item())
                mean_last_losses = np.mean(last_losses)

                ### checkpoint criteria checks and saving
                if checkpoint_inst.check_criteria(mean_last_losses):
                    checkpoint_inst.create_checkpoint(
                        model=model_inst,
                        optimizer=optimizer_inst,
                        scheduler=scheduler_inst,
                        current_iteration=current_iteration,
                        full_config=full_config,
                    )

                scheduler_handler_inst.step(
                    model_inst, optimizer_inst, metric=eval_v_loss
                )

        from IPython import embed

        embed(header="Training ends: Check if everything is as you thought it would be")


if __name__ == "__main__":
    from bbTT.utils.parser import ParserBuilder

    parser = ParserBuilder("tensorboard", "cache")

    main(
        ignore_cache=parser.args.ignore_cache,
        save_cache=parser.args.save_cache,
        tensorboard_name=parser.args.tensorboard_name,
    )
