import awkward as ak

from utils import logger

logger_inst = logger.get_logger(__name__)

def flatten_lists(nested_list):
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten_lists(item)
        else:
            yield item

def depthCount(lst):
    """Takes an arbitrarily nested list as a parameter and returns the maximum depth to which the list has nested sub-lists"""
    if isinstance(lst, list):
        return 1 + max(depthCount(x) for x in lst)
    else:
        return 0

def pad_empty(array: ak.Array, target_length: int, pad_value: float = 0.0):
    """Pads an akward array of lists to a specified target length with a given pad value."""
    array = ak.pad_none(array, target_length, axis=1, clip=True)
    array = ak.fill_none(array, pad_value)
    return array

def find_datasets(dataset_patterns: list[str], year_patterns: list[str], *, file_type: str="root", verbose=True):
    """
    Find all files in variable ${INPUT_DATA_DIR} by using glob patterns following <year_pattern>/<dataset_pattern>/*.<file_type>.
    The result is as a dictionary of form: {dataset_name : [file_paths]}.
    Duplicates in *year_patterns* and *dataset_patterns* will not affect the result.

    Args:
        dataset_patterns (list[str]): Pattern describing our nano AODs, ex: ['hh_ggf_hbb_htt_kl*_kt1*',]
        year_pattern (list[str]): pattern describing the years e.g. "2*pre" -> 22pre, 23pre
        file_type (str, optional): File extension . Defaults to "root".

    Returns:
        dict: dictionary with dataset names as keys and list of file paths as values
    """
    import os
    import pathlib
    # precautions: get dir, wrap strings, set pattern
    logger_inst.info("Start searching for datasets:")
    if (data_dir := os.environ.get("INPUT_DATA_DIR", None)) is None:
        raise ValueError("Environment variable INPUT_DATA_DIR not set! Source setup.sh")

    if isinstance(dataset_patterns, str):
        dataset_patterns = [dataset_patterns]
    if isinstance(year_patterns, str):
        year_patterns = [year_patterns]

    file_pattern = f"*.{file_type}"

    # resolve year pattern and remove duplicates
    years = []
    for year_pattern in year_patterns:
        years += [year.name for year in pathlib.Path(data_dir).glob(f"{year_pattern}")]
    years = sorted(set(years))

    data = {}
    missing = []
    for year in years:
        data[year] = {}
        for dataset_patter in dataset_patterns:
            datasets = list(pathlib.Path(data_dir).glob(f"{year}/{dataset_patter}"))

            if len(datasets) == 0:
                raise ValueError(f"dataset pattern {dataset_patter} for {year} resulted in 0 datasets")

            for dataset in datasets:
                files = sorted(map(str, pathlib.Path(dataset).glob(file_pattern)))
                if len(files) == 0:
                    logger_inst.critical(f"{dataset} has 0 files")
                    missing.append(dataset)
                if verbose:
                    size = round(sum(os.path.getsize(f) for f in files) / (1024**2),2)
                    logger_inst.debug(f"+{len(files)} files | size {size} MB | {year}/{dataset.name}")
                data[year][dataset.name] = files
    if not data:
        raise ValueError("No datasets found with given patterns")

    if missing:
        missing_msg = '\n\t'.join(missing)
        raise ValueError(f"following datasets has 0 files:\n{missing_msg}")
    # merge over years era information is not needed
    merged_over_era_data = {}
    for era in list(data.keys()):
        dataset_dict = data.pop(era)
        for dataset, files in dataset_dict.items():
            if dataset not in merged_over_era_data:
                merged_over_era_data[dataset] = []
            merged_over_era_data[dataset].extend(files)
    return merged_over_era_data
