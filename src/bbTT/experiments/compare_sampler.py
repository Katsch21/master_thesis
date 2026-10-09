from __future__ import annotations

import bbTT.data_handling.sampler as sampler_old

# standard imports
# package imports
import numpy as np
import torch

import bbTT.data_handling.sampling.sampler as sampler
from bbTT.configs.full_config import FullConfig

# personal imports
from bbTT.data_handling import io, k_fold
from bbTT.data_handling.sampling.weight import WeightAggregator

# from .train_utils import log_metrics
from bbTT.monitoring.logger.logger import get_logger

CPU = torch.device("cpu")
CUDA = torch.device("cuda")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
full_config = FullConfig()
torch.manual_seed(full_config.training_config.seed)
np.random.seed(full_config.training_config.seed)


def main(**kwargs):
    # prepare logger
    logger_inst = get_logger(__name__)
    logger_inst.info(f"DEVICE: {DEVICE}")

    for current_fold in full_config.training_config.train_folds:
        logger_inst.info(f"Trainings fold: {current_fold}/{full_config.training_config.k_fold - 1}")

        # prepare old data

        events = io.get_data(
            full_config.dataset_config, ignore_cache=kwargs["ignore_cache"], save_cache=kwargs["save_cache"]
        )
        # split data into training and validation according to fold and get collect all weight statistics
        fold_split_coordinator = k_fold.FoldAndSplitCoordinator(
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
        train_events, validation_events = (
            fold_split_coordinator(events, which="training", columns=columns_to_split),
            fold_split_coordinator(events, which="validation", columns=columns_to_split),
        )  # noqa
        from bbTT.data_handling.preprocessing import WeightAggregator as WA

        wa_inst = WA(events, fold_split_coordinator.indices)

        # release initial fields
        for key in list(events.keys()):
            del events[key]

        logger_inst.info("Start creation of Sampler")

        _sampler_config = {
            "weight_aggregator_inst": wa_inst,
            "target_map": full_config.dataset_config.target_map,
            "min_size": full_config.training_config.min_events_in_batch,
            "batch_size": full_config.training_config.training_batch_size,
            "sample_ratio": full_config.training_config.sample_ratio,
            "sub_sample_ratio": full_config.training_config.sub_process_ratios,
        }

        ts_old = sampler_old.create_sampler(
            train_events,
            train=True,
            **_sampler_config,
        )
        vs_old = sampler_old.create_sampler(
            validation_events,
            train=False,
            **_sampler_config,
        )

        # get weighted mean and std of expected batch composition
        logger_inst.info("Start model building and configuration")

        from bbTT.data_handling.preprocessing import get_batch_statistics_from_sampler_old

        old_mean, old_std = get_batch_statistics_from_sampler_old(
            sampler=ts_old,
            padding_values=full_config.dataset_config.dummy_values,
            features=full_config.dataset_config.continuous_features,
            return_dummy=full_config.debug_config.get_batch_statistic_return_dummy,
        )

        # prepare new data
        events = io.get_data(
            full_config.dataset_config, ignore_cache=kwargs["ignore_cache"], save_cache=kwargs["save_cache"]
        )

        fold_split_coordinator = k_fold.FoldAndSplitCoordinator(
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
        train_events, validation_events = (
            fold_split_coordinator(events, which="training", columns=columns_to_split),
            fold_split_coordinator(events, which="validation", columns=columns_to_split),
        )  # noqa
        weight_aggregator = WeightAggregator(events, fold_split_coordinator.indices)

        # release initial fields
        for key in list(events.keys()):
            del events[key]

        logger_inst.info("Start creation of Sampler")

        _sampler_config = {
            "weight_aggregator_inst": weight_aggregator,
            "target_map": full_config.dataset_config.target_map,
            "min_size": full_config.training_config.min_events_in_batch,
            "batch_size": full_config.training_config.training_batch_size,
            "sample_ratio": full_config.training_config.sample_ratio,
            "sub_sample_ratio": full_config.training_config.sub_process_ratios,
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

        from bbTT.data_handling.preprocessing import get_batch_statistics_from_sampler

        new_mean, new_std = get_batch_statistics_from_sampler(
            sampler=training_sampler,
            padding_values=full_config.dataset_config.dummy_values,
            features=full_config.dataset_config.continuous_features,
            return_dummy=full_config.debug_config.get_batch_statistic_return_dummy,
        )

        from IPython import embed

        embed(
            header="MESSAGE Line 139 | File: /afs/desy.de/user/w/wiedersb/xxl/pytorch_network_playground/src/bbTT/experiments/compare_sampler.py"
        )


if __name__ == "__main__":
    from bbTT.utils.parser import ParserBuilder

    parser = ParserBuilder("tensorboard", "cache")

    main(
        ignore_cache=parser.args.ignore_cache,
        save_cache=parser.args.save_cache,
        tensorboard_name=parser.args.tensorboard_name,
    )
