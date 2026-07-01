from __future__ import annotations

# standard imports
import dataclasses

# package imports
import numpy as np
import torch

# personal imports
from data import cache, load_data, preprocessing, sampler
from loss.utils import init_loss
from models.utils import init_model
from optimizer.early_stopping import CheckPoint
from optimizer.utils import init_optimizer, init_scheduler
from train.loops import TrainingLoop, ValidationLoop
from train.train_config import full_config
from train.train_utils import log_metrics
from utils import logger

CPU = torch.device("cpu")
CUDA = torch.device("cuda")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

torch.manual_seed(full_config.training_config.seed)
np.random.seed(full_config.training_config.seed)

def main(**kwargs):
    # prepare logger
    logger_inst = logger.get_logger(__name__)
    logger_inst.info(f"DEVICE: {DEVICE}")
    tensorboard_writer = logger.TensorboardLogger(
        name=cache.hash_dictionary(dataclasses.asdict(full_config.training_config)),
        path=kwargs["tensorboard_name"],
        )
    # TODO add LOGGER FILEPATH in same directory of tensorboard
    logger_inst.i_info(f"Tensorboard logs: {tensorboard_writer.path}")

    # load data
    for current_fold in (full_config.training_config.train_folds):
        logger_inst.info(f"Trainings fold: {current_fold}/{full_config.training_config.k_fold - 1}")
        #-----
        ### data loading and preprocessing
        #-----
        # HINT: order matters, due to memory constraints views are moved in and out of dictionaries
        # load data from cache is necessary or from root files
        # events is of form : {uid : {"continuous","categorical", "weight": torch tensor}}
        events = load_data.get_data(full_config.dataset_config, ignore_cache=kwargs["ignore_cache"], _save_cache=kwargs["save_cache"])

        # split data into training and validation according to fold and get collect all weight statistics
        fold_split_coordinator = preprocessing.FoldAndSplitCoordinator(
            events=events,
            c_fold=current_fold,
            k_fold=full_config.training_config.k_fold,
            seed=full_config.training_config.seed,
            training_percentage=0.75,
            randomize=True,
        )

        columns_to_split = ("continuous", "categorical", "event_id", "normalization_weights", "product_of_weights", "evaluation_mask")
        train_events, validation_events = fold_split_coordinator(events, which="training", columns=columns_to_split), fold_split_coordinator(events, which="validation", columns=columns_to_split) #noqa
        weight_aggregator = preprocessing.WeightAggregator(events, fold_split_coordinator.indices)

        # release initial fields
        for key in list(events.keys()):
            del events[key]

        logger_inst.info("Start creation of Sampler")

        _sampler_config = {
            "weight_aggregator_inst" : weight_aggregator,
            "target_map" : full_config.dataset_config.target_map,
            "min_size" : full_config.training_config.min_events_in_batch,
            "batch_size" : full_config.training_config.t_batch_size,
            "sample_ratio" : full_config.training_config.sample_ratio,
            "sub_sample_ratio" : full_config.training_config.sub_process_ratios,
        }

        training_sampler = sampler.create_sampler(
            train_events,
            train=True,
            **_sampler_config,
        )
        validation_sampler = sampler.create_sampler(
            validation_events,
            train=False,
            **_sampler_config,
        )
        # share relative weight from training batch statistic to validation sampler
        training_sampler.share_weights_between_sampler(validation_sampler)
        # get weighted mean and std of expected batch composition
        logger_inst.info("Start model building and configuration")

        full_config.model_building_config.mean, full_config.model_building_config.std = preprocessing.get_batch_statistics_from_sampler(
            training_sampler,
            padding_values=full_config.dataset_config.dummy_values,
            features=full_config.dataset_config.continuous_features,
            return_dummy=full_config.debug_config.get_batch_statistic_return_dummy,
        )
        #----
        ### model build and configuration, including optimizer, scheduler, early stopping and loss function
        #----
        model_inst = init_model(full_config=full_config)
        model_inst = model_inst.to(DEVICE).train()

        training_loop, validation_loop = TrainingLoop(full_config), ValidationLoop(full_config)

        optimizer_inst = init_optimizer(full_config=full_config, model_inst=model_inst)
        training_loss_inst, validation_loss_inst = init_loss(full_config=full_config, device=DEVICE, training_sampler=training_sampler)
        scheduler_inst = init_scheduler(full_config=full_config, optimizer_inst=optimizer_inst)
        checkpoint_inst = CheckPoint(checkpoint_name=full_config.training_config.save_model_name, checkpoint_fold=current_fold)

        #----
        ### training loop
        #----
        logger_inst.info("Start training loop")
        for current_iteration in range(1_000_000):
            t_loss = training_loop(
                model = model_inst,
                loss_fn = training_loss_inst,
                optimizer = optimizer_inst,
                sampler = training_sampler,
                device=DEVICE,
                sample_columns = full_config.training_config.sample_attributes,
                scheduler_inst = scheduler_inst,
            )
            if current_iteration % full_config.training_config.verbose_interval == 0:
                tensorboard_writer.log_loss({"batch_loss": t_loss.item()}, step=current_iteration)
                current_lr = optimizer_inst.param_groups[0]["lr"]
                logger_inst.training(f"T-It: {current_iteration} - LR: {current_lr} - batch loss: {t_loss.item():.2E}")
                tensorboard_writer.log_lr(current_lr, step=current_iteration)
            #----
            #### Evaluation of training and validation data, logging and checkpointing
            #----
            if (current_iteration % full_config.training_config.validation_interval == 0) & (current_iteration >= 0):
                # evaluation of training data
                logger_inst.info(f"Iteration {current_iteration}. Start evaluation of training data.")

                eval_t_loss, (eval_t_pred, eval_t_tar, eval_t_weights) = validation_loop(
                    model_inst,
                    validation_loss_inst,
                    training_sampler,
                    sample_columns=full_config.training_config.sample_attributes,
                    device=DEVICE
                    )
                # TODO when edges should be tracked add this in a way that is universal and does not break for models without binning layer, e.g. add property to model that returns None if no binning layer is present and add check in log_metrics
                # evaluation of validation
                logger_inst.info(f"Iteration {current_iteration}. Start evaluation of validation data.")

                eval_v_loss, (eval_v_pred, eval_v_tar, eval_v_weights) = validation_loop(
                    model_inst,
                    validation_loss_inst,
                    validation_sampler,
                    sample_columns=full_config.training_config.sample_attributes,
                    device=DEVICE
                    )
                # TODO when edges should be tracked add this in a way that is universal and does not break for models without binning layer, e.g. add property to model that returns None if no binning layer is present and add check in log_metrics
                if full_config.training_config.log_metrics:
                    log_metrics(
                        tensorboard_inst = tensorboard_writer,
                        iteration_step = current_iteration,
                        sampler_output = (eval_t_pred, eval_t_tar, eval_t_weights),
                        target_map = full_config.dataset_config.target_map,
                        mode = "train",
                        loss = eval_t_loss.item(),
                        lr = optimizer_inst.param_groups[0]["lr"],
                        # binning_edges = model_inst.binning_layer.edges.detach().cpu(),
                        # binning_edges = ,
                        current_iteration = current_iteration,
                        # kernels = model_inst.binning_layer.kernels,
                    )



                    log_metrics(
                        tensorboard_inst = tensorboard_writer,
                        iteration_step = current_iteration,
                        sampler_output = (eval_v_pred, eval_v_tar, eval_v_weights),
                        target_map = full_config.dataset_config.target_map,
                        mode = "validation",
                        loss = eval_v_loss.item(),
                        # TODO binning edges and kernels are only defined for BinnedLBN make universal
                        # binning_edges = model_inst.binning_layer.edges.detach().cpu(),
                        binning_edges = full_config.binning_config.num_bins,
                        current_iteration = current_iteration,
                        # kernels=model_inst.binning_layer.kernels,
                    )
                logger_inst.training(f"Iteration: {current_iteration} - TLoss: {eval_t_loss:.2E} VLoss: {eval_v_loss:.2E}")


                ### checkpoint criteria checks and saving
                if checkpoint_inst.check_criteria(eval_v_loss):
                    checkpoint_inst.create_checkpoint(
                        model=model_inst,
                        optimizer=optimizer_inst,
                        scheduler=scheduler_inst,
                        current_iteration=current_iteration,
                        full_config=full_config,
                    )

                # if metric does not improve x-times reduce
                # load previous checkpoint with reduces learning rate

                # when using ReduceLROnPlateau, step scheduler and check if lr was reduced, if yes load last checkpoint and update optimizer with new lr
                if isinstance(scheduler_inst, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    previous_lr =  optimizer_inst.param_groups[0]["lr"]
                    scheduler_inst.step(eval_v_loss)
                    current_lr = optimizer_inst.param_groups[0]["lr"]
                    if previous_lr != current_lr:
                        logger_inst.info(
                            f"{previous_lr} -> {current_lr}" +
                            "\nReload model and optimizer from iteration"
                            f" {checkpoint_inst.last_checkpoint['iteration']}")

                        model_inst.load_state_dict(checkpoint_inst.last_checkpoint["model_state_dict"])
                        optimizer_inst.load_state_dict(checkpoint_inst.last_checkpoint["optimizer_state_dict"])
                        # since scheduler and optimizer are coupled
                        # overwrite old lr in checkpoint with lr after scheduler step
                        optimizer_inst.param_groups[0]["lr"] = current_lr

        from IPython import embed
        embed(header="Training ends: Check if everything is as you thought it would be")

if __name__ == "__main__":
    from utils.parser import ParserBuilder
    parser = ParserBuilder("tensorboard", "cache")

    # call main with parsed args
    main(
        ignore_cache=parser.args.ignore_cache,
        save_cache=parser.args.save_cache,
        tensorboard_name=parser.args.tensorboard_name

        )
