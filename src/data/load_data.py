
import pathlib
from typing import Union

import awkward as ak
import numpy as np
import torch
import uproot
import numpy.lib.recfunctions as rfn

from data.cache import DataCacher
from data.utils import depthCount
from utils import logger

logger_inst = logger.get_logger(__name__)

def root_to_numpy(
    files_path: Union[list[str],str],
    branches: Union[list[str], str, None]=None,
    cut: list[str]=None,
    ) -> ak.Array:
    """
    Load all root files in *files_path* and return them as a single awkward array.
    If only certain branches are needed, they can be specified in *branches*.
    To prevent loading unnecessary data a list of *cut*s can be added.

    Args:
        files_path (list[str], str): list of root files or single root file
        branches (list[str], str, optional): branches that should be loaded e.g. ["events", "run"]. If None loads all branches . Defaults to None.

    Returns:
        ak.Array: awkward array containing all data from the root files

    """
    logger_inst.info("Start loading and conversion of root files:")
    if depthCount(branches) > 1 and branches is not None:
        raise ValueError(f"branches must be a flat list but is {depthCount(branches)}-dimensional")

    # set of branches that are always extracted from root files
    # purpose -> fields:
    #   process_id -> filtering of subphasespaces
    #   normalization_weight -> used as sample weight for batch
    #   used for baseline cut -> tau2_isolated, lepton_os, channel_id
    #   event number used for k-fold splitting -> event
    #   oversampling weight that defines the fraction within batch -> normalization_weight
    meta_fields = {"process_id", "tau2_isolated", "leptons_os", "channel_id", "event", "normalization_weight"}

    # training and evaluation phase space are not the same
    # a transfer weight can be calculated using the product of these weights
    weights = {
    "normalized_pdf_weight","normalized_murmuf_weight","normalized_pu_weight",
    "normalized_isr_weight","normalized_fsr_weight","normalized_njet_btag_weight_pnet",
    "electron_id_weight","electron_reco_weight","muon_id_weight","muon_iso_weight",
    "tau_weight","trigger_weight", "dy_weight","top_pt_weight"
    }
    all_branches = set(branches).union(meta_fields)

    # handling cuts
    # by default all analysis has a base cut applied
    # if further cuts are desired, cut is applied on top of baseline cut

    baseline_cuts = [
        "(tau2_isolated == 1)",
        "(leptons_os == 1)",
        "((channel_id == 1) | (channel_id == 2) | (channel_id == 3))",
        "(reg_dnn_moe_vis_tau2_charge == 1) | (reg_dnn_moe_vis_tau2_charge == -1)"
    ]

    if isinstance(cut, str):
        cut = [cut]
    if cut is None:
        cut = []
    final_cut = "&".join(baseline_cuts + cut)

    # load root files and combine the array to continuous arrays
    if isinstance(files_path, str):
        files_path = [files_path]

    arrays = []
    for file_path in files_path:
        logger_inst.debug(f"loading: {file_path}")
        with uproot.open(file_path, object_cache=None, array_cache=None) as file:
            # loading of data is split into 3 steps: all_common, weights, evaluation phasespace
            # reason for 2 is that weights are not present in all datasets
            # e.x. muon_id_weight does not exist in dy datasets, thus require filtering
            # reason for 3: masks that are necessary later are calculated here
            tree = file["events"]

            # step 1
            all_branches_array = tree.arrays(all_branches, library="ak", cut=final_cut)

            # step 2
            weights_in_root_file = set(tree.keys()).intersection(weights)
            weights_arrays = tree.arrays(weights_in_root_file, library="ak", cut=final_cut)

            combined_weight = 1
            for weight in weights_in_root_file:
                combined_weight = combined_weight * weights_arrays[weight]
            all_branches_array["combined_weight"] = combined_weight

            # step 3
            di_tau_mask, di_bjet_mask, bjet_mask = res1b_and_res2b_phase_space_mask(
                uproot_file=tree,
                year=pathlib.Path(file_path).parents[1].stem,
                cut=final_cut,
                suffix=("res_dnn_pnet" if branches[0].startswith("res_dnn_pnet") else "reg_dnn_moe"))
            all_branches_array["bjet_mask"] = bjet_mask
            all_branches_array["di_tau_mask"] = di_tau_mask
            all_branches_array["di_bjet_mask"] = di_bjet_mask

            arrays.append(all_branches_array.to_numpy())
    arrays = np.concatenate(arrays, axis=0)
    return arrays


def res1b_and_res2b_phase_space_mask(uproot_file: str, year: list[str], cut: list[str], suffix: str="res_dnn_pnet"):
    """
    Calculates Masks to get into our evaluation phase space.
    Open *uproot_file* by specific *year*, apply  base *cut* and depending on the producer add a *suffix* to fields in root file.
    Definition of mask is defined in https://github.com/uhh-cms/hh2bbtautau/blob/master/hbt/categorization/default.py#L206-L240

    Args:
        uproot_file (str): Uproot opened root file
        year (list[str]): Year string e.g. "22pre"
        cut (list[str]): Base cut to be applied, should match the one used in root_to_numpy
        suffix (str, optional): Suffix for fields in uproot file. Defaults to "res_dnn_pnet".

    Returns:
        ak.array: Masks for di_tau_mass_window, di_bjet_mass_window, bjet
    """
    # as taken from https://github.com/uhh-cms/hh2bbtautau/blob/master/hbt/config/configs_hbt.py#L1252
    def particle_net_wp(year, wp_level="medium"):
        particle_net_wp = {
            "loose": {"22pre": 0.047, "22post": 0.0499, "23pre": 0.0358, "23post": 0.0359, "2024": None}[year],
            "medium": {"22pre": 0.245, "22post": 0.2605, "23pre": 0.1917, "23post": 0.1919, "2024": None}[year],
            "tight": {"22pre": 0.6734, "22post": 0.6915, "23pre": 0.6172, "23post": 0.6133, "2024": None}[year],
            "xtight": {"22pre": 0.7862, "22post": 0.8033, "23pre": 0.7515, "23post": 0.7544, "2024": None}[year],
            "xxtight": {"22pre": 0.961, "22post": 0.9664, "23pre": 0.9659, "23post": 0.9688, "2024": None}[year],
        }
        return particle_net_wp[wp_level]
    # load the necessary events with applied cut from baseselection
    # all particles are pre rotate relative to visible tau system
    b_tag_wp = particle_net_wp(year, "medium")
    leptons_fields = [f"{suffix}_vis_tau{num}_{kin}" for num in ("1", "2") for kin in ("px", "py", "pz", "e")]
    hhbjet_fields = ["HHBJet_mass", "HHBJet_btagPNetB"]
    di_bjet_fields = [f"{suffix}_bjet{num}_{kin}" for num in ("1", "2") for kin in ("px", "py", "pz", "e")]

    events = uproot_file.arrays(leptons_fields + hhbjet_fields + di_bjet_fields, library="ak", cut=cut)
    ### masks
    # tau mass window
    l_px = events[f"{suffix}_vis_tau1_px"] + events[f"{suffix}_vis_tau2_px"]
    l_py = events[f"{suffix}_vis_tau1_py"] + events[f"{suffix}_vis_tau2_py"]
    l_pz = events[f"{suffix}_vis_tau1_pz"] + events[f"{suffix}_vis_tau2_pz"]
    l_e = events[f"{suffix}_vis_tau1_e"] + events[f"{suffix}_vis_tau2_e"]

    # since no coffee behavior, calculate mass by manually from 4 vector
    di_tau_mass = (l_e**2 - (l_px**2 + l_py**2 + l_pz**2))**0.5
    di_tau_mass_window_mask = (
        (di_tau_mass >= 15) &
        (di_tau_mass <= 130)
    )

    # have atleast 1 bjet
    bjet_mask = ak.sum(events.HHBJet_btagPNetB > b_tag_wp, axis=1) >= 1

    # wrong
    # di_bjet_mass = ak.sum(events.HHBJet_mass, axis=1)

    b_px = events[f"{suffix}_bjet1_px"] + events[f"{suffix}_bjet2_px"]
    b_py = events[f"{suffix}_bjet1_py"] + events[f"{suffix}_bjet2_py"]
    b_pz = events[f"{suffix}_bjet1_pz"] + events[f"{suffix}_bjet2_pz"]
    b_e = events[f"{suffix}_bjet1_e"] + events[f"{suffix}_bjet2_e"]
    di_bjet_mass = (b_e**2 - (b_px**2 + b_py**2 + b_pz**2))**0.5

    di_bjet_mass_window_mask = (
        (di_bjet_mass >= 40) &
        (di_bjet_mass <= 270)
    )

    return di_tau_mass_window_mask, di_bjet_mass_window_mask, bjet_mask


def load_data(datasets, columns: Union[list[str],str, None]=None, cuts: Union[list[str], None]=None):
    """
    Loads data with given *file_type* in given a *dataset_pattern* and *year_pattern*. If only certain columns are needed, they can be specified in *columns*.
    The data sorted by year and dataset name is returned as a nested dictionary in awkward format.

    Args:
        dataset_patter (str): pattern describing dataset e.g. "dy_*" -> all drell-yan datasets
        columns (list[str], str, optional): columns that should be lodead e.g. ["events", "run"]. If None loads all columns . Defaults to None.
        cuts (list[str]): list of cuts to be applied on top of baseline cut, which are defined in root_to_numpy. Defaults to None.

    Returns:
        dict: {year:{pid: List(Ids)}}
    """
    def load_data_per_process_id(dataset_paths, branches, cut=None):
        # helper to load root data into a dictionary of form:
        # {year:{dataset : array}}, where array is a structured array
        def sort_by_process_id(array):
            # helper to extract data by process id and group them by this
            pids = array["process_id"]
            unique_ids = np.unique(pids)
            p_array = {}
            for uid in unique_ids:
                mask = pids == uid
                p_array[int(uid)] = array[mask]
            return tuple(p_array.items())
        data = {}
        for dataset, files in dataset_paths.items():
            events = root_to_numpy(files, branches=branches, cut=cut)
            p_arrays = sort_by_process_id(events)
            for pid, p_array in p_arrays:
                uid = (dataset[:2],pid)
                if uid not in data:
                    data[uid] = []
                data[uid].append(p_array)
                logger_inst.debug(f"{dataset} | PID: {pid} | {len(p_array)}")
        return data

    def merge_per_pid(data):
        # helper to merge all process_ids together to continuous array
        logger_inst.info("Start merging arrays per process id")
        keys = list(data.keys())

        for i, uid in enumerate(keys):
            logger_inst.debug(f"{i}/{len(keys)}: {uid}")
            arrays = data.pop(uid)
            concat = np.concatenate(arrays, axis=0)
            data[uid] = concat
        return data
    data = load_data_per_process_id(datasets, branches=list(columns), cut=cuts)
    data = merge_per_pid(data)
    return data


def handle_weights_and_convert_to_torch(events: np.array, continuous_features: list[str], categorical_features: list[str]):
    """
    Calculates final weights, extract masks aswell as extract all *continuous_features* and *categorical_features* from structured numpy array *events*.
    Converts all arrays to torch tensors and returns a dictionary containing these.

    Args:
        events (np.array): _description_
        continuous_features (list[str]): List of continuous features to be extracted
        categorical_features (list[str]): List of categorical features to be extracted
        dtype (torch.dtype, optional): Torch dtype. Defaults to None.
    """

    def filter_nan_mask(array, features, uid):
        event_mask = np.zeros(array.size, dtype=np.bool)
        for f in features:
            event_mask |= np.isnan(array[f])
        num_filter = np.sum(event_mask)
        if num_filter:
            logger_inst.warning(f"Filtered {num_filter} Nan events from pid {uid}")
        return ~event_mask

    for uid in list(events.keys()):
        arr = events.pop(uid)

        # filter all nans out
        event_mask = filter_nan_mask(arr, continuous_features + categorical_features, uid)
        arr = arr[event_mask]

        # if resulting tensor is empty just skip
        if arr.size == 0:
            logger_inst.warning(f"Skipping {uid} due to zero elements - which can happen after filtering nans")
            continue

        # handle non feature columns first:

        continuous_tensor = struct_to_group_tensor(arr, continuous_features, dtype=torch.float32)
        categorical_tensor = struct_to_group_tensor(arr, categorical_features, dtype=torch.float32)

        # handling weights and convert to torch tensors
        # single numbers cant be converted by using from_numpy thus have to be wrapped in array
        masks_tensor = struct_to_group_tensor(arr, ["bjet_mask", "di_tau_mask", "di_bjet_mask"], torch.bool)
        final_mask = masks_tensor[:, 0] & masks_tensor[:, 1] & masks_tensor[:, 2]
        # total_bjet_weight = torch.tensor(np.sum(arr["combined_weight"][arr["bjet_mask"]]))
        # total_di_tau_weight = torch.tensor(np.sum(arr["combined_weight"][arr["di_tau_mask"]]))
        # total_di_bjet_weight = torch.tensor(np.sum(arr["combined_weight"][arr["di_bjet_mask"]]))


        # some arrays have negative strides for some reason, which torch cannot handle -> cast to contiguous array first
        weights_tensor = struct_to_group_tensor(arr, ["normalization_weight", "combined_weight"], dtype=torch.float32)
        normalization_weights = weights_tensor[:, 0]
        sum_of_normalization_weights = torch.sum(normalization_weights)

        product_of_all_weights = weights_tensor[:, 1]
        sum_of_combined_weights = torch.sum(product_of_all_weights)
        total_evaluation_weight = torch.sum(product_of_all_weights[final_mask])

        event_id = struct_to_group_tensor(arr, ["event"], dtype=torch.int64).flatten()

        events[uid] = {
            "continuous": continuous_tensor,
            "categorical": categorical_tensor,
            "event_id" : event_id,
            "normalization_weights" : normalization_weights,
            "product_of_weights" : product_of_all_weights,

            "total_product_of_weights" : sum_of_combined_weights,
            "total_normalization_weights" : sum_of_normalization_weights,

            "total_evaluation_weight" : total_evaluation_weight,
            "evaluation_mask": final_mask,
            "mask" : {
                "bjet": masks_tensor[:, 0],
                "di_tau": masks_tensor[:, 1],
                "di_bjet": masks_tensor[:, 2],
                },
        }
        del arr
    return events



def get_data(config , _save_cache = False, ignore_cache=False) -> dict[torch.Tensor]:
    """
    Main function to combine all steps from loading root files to filter by process ids and finally convert to torch

    Args:
        config (DataClass): Config as defined in train_config.py
        _save_cache (bool, optional): Save the result of this function as cache, using config as hash. Defaults to False.
        ignore_cache (bool, optional): Rerun loading of data if true, ignoring existing cache. Defaults to False.

    Returns:
        dict[torch.Tensor]: Dictionary with torch tensors
    """
    cacher = DataCacher(config=config)

    # when cache exist load -> it and return the data
    if not ignore_cache and cacher.path.exists():
        events = cacher.load_cache()
    else:
        logger_inst.info("Start loading and filtering of data")
        events = load_data(
            config.datasets,
            columns=config.continuous_features + config.categorical_features,
            cuts=config.cuts,
        )
        logger_inst.info("Start handling weights and conversion to torch tensors")
        events = handle_weights_and_convert_to_torch(
            events=events,
            continuous_features=config.continuous_features,
            categorical_features=config.categorical_features,
        )
        logger_inst.info("Done prepareing data")
        if _save_cache:
            try:
                cacher.save_cache(events)
            except Exception as e:
                from IPython import embed
                embed(header=f"{e}\n Saving Cache did not work out - going debugging to manually save \'events\' with \'cacher.save_cache\'")
    return events

def struct_to_group_tensor(arr: np.typing.NDArray, fields: tuple[str], dtype: torch.dtype=torch.float32):
    """
    Small helper convert struct *arr* *fields* into torch tensor of given *dtype*.

    Args:
        arr (np.typing.NDArray): structured array
        fields (tuple[str]): fields one wants to extract
        dtype (torch.dtype, optional): final dtype. Defaults to torch.float32.

    Returns:
        torch.Tensor: Tensor of extracted fields in given dtype
    """
    # get numpy equivalent dtype
    np_dtype = torch.empty(0, dtype=dtype).numpy().dtype
    dense = rfn.structured_to_unstructured(arr[fields], dtype=np_dtype)
    # some arrays have negative strides for some reason, which torch cannot handle -> cast to contiguous array first
    dense = np.ascontiguousarray(dense)
    return torch.from_numpy(dense)
