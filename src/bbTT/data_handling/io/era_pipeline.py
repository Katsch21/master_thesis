
import torch

from bbTT.data_handling.cache import DataCacher
from bbTT.data_handling.io.era_merging import merge_era_events
from bbTT.data_handling.io.find_datasets import structure_datasets_after_eras
from bbTT.data_handling.io.tensor_conversion import filter_and_convert_to_torch
from bbTT.data_handling.io.uid_streaming import stream_events_by_uid
from bbTT.monitoring.logger.logger import get_logger

logger_inst = get_logger(__name__)

def create_era_caches(config, cache: DataCacher, ignore_cache: bool, save_cache: bool) -> None:
    """
    Ensure every configured era exists in cache, loading + processing from raw files
    where needed. Does not merge or return any data — purely a cache-population step.
    """
    era_datasets = structure_datasets_after_eras(config)
    for era in config.eras:
        if not ignore_cache and cache.era_exists(era):
            logger_inst.info(f"Cache already present for era {era}")
            continue

        era_events, n_per_dataset, n_per_pid = _load_and_process_era(
            config,
            datasets_per_era=era_datasets,
            era=era,
        )

        cache.save_sizes(sizes=n_per_dataset, era=era, is_pid=False)
        cache.save_sizes(sizes=n_per_pid, era=era, is_pid=True)
        _cache_era(cache, era, era_events, save_cache)
        del era_events


def _load_and_process_era(config, datasets_per_era, era):
    """
    Load one era from raw files, filter, and convert to torch tensors.
    """
    logger_inst.info(f"Start loading and filtering of data for era {era}")

    era_datasets = datasets_per_era[era]
    era_events, num_events_per_dataset, num_events_per_pid = stream_events_by_uid(
        era_datasets,
        columns=config.uproot_continuous_columns + config.uproot_categorical_columns,
        cut=config.uproot_cuts,
        flush_threshold_rows=config.flush_threshold,
    )

    logger_inst.info("Start handling weights and conversion to torch tensors")
    era_events = filter_and_convert_to_torch(
        events=era_events,
        continuous_features=config.uproot_continuous_columns,
        categorical_features=config.uproot_categorical_columns,
    )
    return era_events, num_events_per_dataset, num_events_per_pid


def _cache_era(cache: DataCacher, era, era_events, save_cache: bool) -> None:
    """
    Persist one era's data to cache. Always writes internally so downstream
    merging can stream from disk in sorted order rather than holding everything
    in memory; if save_cache is False, mark it for cleanup after use.
    """
    try:
        cache.save_era(era=era, events=era_events)
        logger_inst.info(f"Finished cache for era {era}")
    except Exception as e:
        logger_inst.exception(f"Saving cache for era {era} failed")
        from IPython import embed
        embed(
            header=f"{e}\nSaving Cache did not work out - going debugging to manually save 'events' with 'cacher.save_era'"
        )


def load_and_merge_eras(config, cache: DataCacher) -> dict[str, torch.Tensor]:
    """
    Load every configured era from cache, smallest first, merging incrementally
    to minimize peak memory during concatenation. Assumes all eras are already cached.
    """

    def sort_key(era):
        total_size = sum(cache.load_era_sizes(era).values())
        return total_size

    sorted_eras = sorted(config.eras, key=sort_key)

    all_events = {}
    for era in sorted_eras:
        logger_inst.info(f"Loading cached data for era {era}")
        era_events = cache.load_era(era)
        all_events = merge_era_events(all_events, era_events)
        del era_events

    return all_events


def get_data(config=None, save_cache=False, ignore_cache=False, _hash=None) -> dict[str, torch.Tensor]:
    """
    Main function to combine all steps from loading root files to filter by
    process ids and finally convert to torch.
    """
    if config is not None and _hash is None:
        cache = DataCacher(config=config)
        create_era_caches(config=config, cache=cache, save_cache=save_cache, ignore_cache=ignore_cache)
        return load_and_merge_eras(config=config, cache=cache)
