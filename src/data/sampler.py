from collections import defaultdict

import torch
import torch.utils.data as t_data

from utils import logger
from utils.utils import CPU_DEVICE

logger_inst = logger.get_logger(__name__)

class Process(t_data.Dataset):

    def __init__(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        target: torch.Tensor,
        normalization_weights: torch.Tensor,
        product_of_weights: torch.Tensor,
        weights_statistics: dict[str, torch.Tensor],
        evaluation_space_mask: torch.Tensor,
        process_id: str=None,
        process_type: str=None,
        randomize: bool =True,
    ):
        """
        A Process Dataset class represents all data connected to a process identified by *process id*.
        Each process is also associated with a *process type* like "tt", "dy, "hh" etc.
        The randomize flag controls whether the data is shuffled at the beginning of each full pass.
        The process data itself is stored as torch tensors in the attributes *continuous*, *categorical* and *targets*.
        A process, knows about its individual *normalization_weights* and *product_of_weights*, but also the total weight of the process type.

        The normalization weights represent the scaling from oversampling of the MC generator.
        The product of weights
        The data is stored as tensors and is hold the actual data and weights corresponding to a process id and also all necessary meta information.


        The *product_of_weights* is the product of all event weights assigned during generation and reconstruction.

        Args:
            continuous (torch.Tensor): Tensor with continuous features of shape [num_events, num_continuous_features]
            categorical (torch.Tensor): Tensor with categorical features of shape [num_events, num_categorical_features]
            target (torch.Tensor): Tensor with target one-hot encoded of shape [num_events, num_classes]
            normalization_weights (torch.Tensor): Tensor with normalization weights per event of shape [num_events].
            product_of_weights (torch.Tensor): Tensor of product of weights per event of shape [num_events].
            weight_statitics (dict[str, torch.Tensor]): Dictionary with summary statistics of the weights for the whole subprocess.
            evaluation_space_mask (torch.Tensor): Bool Tensor to bring events into evaluation phase space
            process_id (int, optional): Unique process id of the process. Defaults to None.
            process_type (str, optional): To which process type does the process belong to (ex. "hh"). Defaults to None.
            randomize (bool, optional): Defines if a new cycle of a process instance is shuffled. Defaults to True.
        """

        self.continuous = continuous.to(torch.float32)
        self.categorical = categorical.to(torch.int32)
        self.targets = target.to(torch.float32)

        # product of all weights assigned during generation and reconstruction
        # necessary to transfer mc to real yield (real events one would see in data)

        self.product_of_weights = product_of_weights.to(torch.float32)

        # mc sampler are generated with higher lumi (to reduce stat. unc.)
        # normalization weights scale down to correct lumi
        self.normalization_weights = normalization_weights.to(torch.float32)

        # contains summary of normalization and product of weights for the whole process
        # but also for evaluation, training ,validation phase space
        self.weights_statistics = weights_statistics

        self.evaluation_space_mask = evaluation_space_mask.to(torch.bool)

        self.process_id = process_id
        self.randomize = randomize
        # initialize indices and current idx
        self.reset()
        self.process_type = process_type
        self.sample_size = len(self)
        self.relative_weight = None

    @property
    def uid(self):
        return (self.process_type, self.process_id)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        return self.continuous[idx], self.continuous[idx], self.targets[idx]

    def reset(self):
        """
        Resets the index of the sampler, and randomize the indices for the next sample
        """
        self.current_idx = 0
        if self.randomize:
            self.indices = torch.randperm(len(self))
        else:
            self.indices = torch.arange(len(self))

    def sample(self,sample_from: tuple[str], number: str=None, device=CPU_DEVICE) -> dict[torch.Tensor]:
        """
        Sample *number* of events from attributes defined by *sample_from*.
        Samples from the start again, if maximum number of samples is reached.
        Specified tensors are moved to *device* before returning.

        Args:
            sample_from (tuple[str]): Tuple of strings defining which attributes to sample from.
            number (str, optional): Number of events to sample. If None, sample_size of the process instance is used. Defaults to None.
            device (torch.device, optional): Device to which tensors are moved before returning. Defaults to torch.device("cpu").

        returns: (dict[torch.Tensor]): Dictionary with sampled tensors for each attribute defined in *sample_from*.
        """
        if self.current_idx >= len(self):
            self.reset()

        if number is None:
            number = self.sample_size

        if number > len(self):
            # when required number is too big, set it to len of self to return all data
            number = len(self)

        start_idx = self.current_idx
        # do not drop last, but instead create smaller batch
        next_idx = min(self.current_idx + number, len(self))
        idx = self.indices[start_idx:next_idx]
        self.current_idx = next_idx

        # sample events from sample_from, and normalization weight per default
        sampled_events = {attribute:getattr(self, attribute)[idx].to(device) for attribute in sample_from}
        sampled_events["sample_weights"] = torch.full((len(idx), 1), self.weights_statistics["normalization_weights"]["whole_sum"] / self.sample_size).to(device)
        return sampled_events

        # return self.continuous[idx].to(device), self.categorical[idx].to(device), self.targets[idx].to(device)

    def create_sample_generator(self, sample_from, batch_size=-1, device=CPU_DEVICE):
        """
        Creates a generator object that returns a dict of torch tensors, defined by *sample_from*.
        If *batch
        If *batch_size* is returned as None, the whole underlying data is returned at once, else only a small batch
        The generator does not randomize the underlying data. The generator and sample methods can be used
        independently.

        Args:
            batch_size (int, optional): Sample size per iteration, if None all data is returned at once. Defaults to None.
            device (torch.device, optional): Device to which tensors are moved before yielding. Defaults to torch.device("cpu").

        Yields:
            tuple (torch.Tensor): Tuple with 3 torch tensors for categorical, continuous and target data of the shape [batch_size, num_features]
        """
        # save current state of the sampler to restore after full iteration
        # HINT: Do not mix sample with batch generator, as both manipulate current_idx
        start_idx, next_idx = 0, 0
        indices = torch.arange(len(self))

        if batch_size == -1:
            batch_size = len(self)

        while next_idx < len(self):
            # handle windowing of indices
            next_idx = min(start_idx + batch_size, len(self))
            idx = indices[start_idx:next_idx]
            start_idx = next_idx
            # reset if we reached the end of the dataset and drop last incomplete batch
            sampled_events = {attribute:getattr(self, attribute)[idx].to(device) for attribute in sample_from}
            sampled_events["sample_weights"] = torch.full((len(idx), 1), self.weights_statistics["normalization_weights"]["whole_sum"] / self.sample_size).to(device)
            yield sampled_events


class ProcessSampler(t_data.Sampler):
    signal_background_map = {"hh": "signal", "tt":"background", "dy":"background"}

    def __init__(
        self,
        batch_size: int=1,
        sample_ratio: dict[float]=None,
        sub_sample_ratio: dict[float] = None,
        min_size: int=0,
        target_map: dict[int] = None,
        weight_aggregator_inst = None,
        ):
        """
        Manager for Process instances to regulate sampling across multiple processes.
        Each process introduces own sampling method, which the ProcessSampler accesses.
        The result it put together as dictionary of multiple torch Tensors.
        The number of events in a tensor per sampler are determined by *batch_size*.
        The composition of the batch is determined in 2 stages:
            1. The desired number of events per process type is calculated according to *sample_ratio*. Needs to be sum to 1.0.
            2. The number of events per process id within a process type is calculated according to the total normalization weights of each process.
        To ensure that each process contributes enough events to the batch, a *min_size* per process id can be defined.
        The *target_map* maps process type to the index in the target tensor.

        Args:
            batch_size (int, optional): The number of events in a tensor per sample. Defaults to 1.
            sample_ratio (dict[int], optional): The contribution of each process type in the batch. Defaults to {"dy": 0.25, "tt": 0.25, "hh": 0.5}.
            min_size (int, optional): The minimum number of events per Process. Defaults to 0.
            target_map (dict[int], optional): Mapping from process type to target index. Defaults to {"hh": 0, "tt": 1, "dy": 2}.
        """
        self.process_inst = defaultdict(dict)  # {process_type: {process_id: dataset}}
        # statistics describing whole population of a process type
        # self.total_normalization_weight_per_process_type = {"dy":0, "tt":0, "hh":0}
        # self.total_product_of_weights_per_process_type = {"dy":0, "tt":0, "hh":0}

        self.batch_size = torch.tensor([batch_size], dtype=torch.int32)
        self.sample_ratio = {"dy": 0.25, "tt": 0.25, "hh": 0.5} if sample_ratio is None else sample_ratio
        self.sub_sample_ratio = sub_sample_ratio
        self.min_size = min_size
        self.target_map = {"hh": 0, "tt": 1, "dy": 2} if target_map is None else target_map
        self.weights_aggregator_inst = weight_aggregator_inst

        # set attributes to access dataset properties directly from sampler, enabling dot notation
        self.set_attr([
            # "total_normalization_weights", "total_product_of_weights",
            "relative_weight", "sample_size",
            "process_type", "create_sample_generator", "evaluation_space_mask",
            "total_process_eval_product_of_weight", "total_process_product_of_weight",
            "continuous", "categorical", "targets", "normalization_weights", "product_of_weights",
            ])

    def __getitem__(self, uid):
        return self.processes()[uid]

    @property
    def sampler_total_eval_weight_per_process_type(self) -> dict[torch.Tensor]:
        if hasattr(self, "_sampler_total_eval_weight_per_process_type"):
            return self._sampler_total_eval_weight_per_process_type
        process_type_weight = {"dy":0, "tt":0, "hh":0}
        for pid, value in self.total_process_eval_product_of_weight().items():
            if pid >50000:
                process_type_weight["dy"] += value
            elif pid < 40000 and value > 5000:
                process_type_weight["hh"] += value
            else:
                process_type_weight["tt"] += value
        self._sampler_total_eval_weight_per_process_type = process_type_weight
        return self._sampler_total_eval_weight_per_process_type

    @property
    def sampler_total_weight_per_process_type(self) -> dict[torch.Tensor]:
        """
        Calculates the total product of weights for a whole process type, but only for the data of the sampler.

        Returns:
            dict[torch.Tensor]: Dictionary of all process types with their total product of weights for the sampler
        """
        if hasattr(self, "_sampler_total_weight_per_process_type"):
            return self._sampler_total_weight_per_process_type
        process_type_weight = {"dy":0, "tt":0, "hh":0}
        for pid, value in self.total_process_product_of_weight().items():
            if pid >50000:
                process_type_weight["dy"] += value
            elif pid < 40000 and value > 5000:
                process_type_weight["hh"] += value
            else:
                process_type_weight["tt"] += value
        self._sampler_total_weight_per_process_type = process_type_weight
        return self._sampler_total_weight_per_process_type

    def processes(self) -> dict[Process]:
        """
        By default process instances are nested as "process_type": {uid : process_inst}.
        This returns a flat version removing outer nesting

        Returns:
            dict[Process] : Dictionary of all process instances managed by the sampler
        """
        datasets = {}
        for _, uid_ds in self.process_inst.items():
            datasets.update(uid_ds)
        return datasets

    def apply_func_on_datasets(self, func):
        # small helper to apply function of managet datasets
        return {uid: func(ds) for uid,ds in self.processes().items()}

    def get_attribute_of_datasets(self, attr):
        # small helper to getattributes from managet datasets
        return {uid: getattr(ds, attr) for uid,ds in self.processes().items()}

    def events_per_dataset(self):
        return self.apply_func_on_datasets(len)

    def __len__(self):
        return sum(list(self.events_per_dataset().values()))

    def add_process_instance(self, process_inst: Process):
        """
        Adds a *process_inst* to the collection of the sampler and update statitics related to it.

        Args:
            process_inst (Process): Instance of the Dataset class
        """
        self.process_inst[process_inst.process_type][process_inst.process_id] = process_inst

    @property
    def keys(self):
        return list(self.process_inst.keys())

    def print_sample_rates(self):
        """
        Function to print sample rates of
        """
        msg = {}
        sum_batch = {}
        for dataset_type, uid in self.process_inst.items():
            msg[dataset_type] = []
            _sum_batch = 0
            for (pid), ds in uid.items():
                msg[dataset_type].append(f"{pid} | {ds.sample_size}")
                _sum_batch += ds.sample_size

            sum_batch[dataset_type] = (_sum_batch, self.sample_ratio[dataset_type] * self.batch_size.item())

        # print all processes with their sample size
        _msg = "\n".join([
            f"{dataset_type.upper()}:\n\t" + "\n  ".join(uid_with_samples)
            for dataset_type, uid_with_samples in msg.items()
            ])
        logger_inst.debug(
            "Sample rates for Dataset of Era (PID | Sample Rate)\n" +
            _msg
        )
        # print summary statistic per dataset type
        _msg = "\n".join([
            f"{dataset_type}: {sum_of_batch} | {expected_batch_size}"
            for dataset_type, (sum_of_batch, expected_batch_size) in self.process_inst.items()

        ])
        logger_inst.debug(
            "Summary statitics:\n" +
            _msg
        )


    def calculate_sample_size(self, process_type: str):
        """
        Calculates the expected Sample Size for each Process ID within a *process_type*.

        How the algorithm works in detail:
        For example we can have 3 process ID {hh: 1/3, tt: 1/3, dy: 1/3}, thus 1/3 of a batch total size is dy.
        Dy is build of many subprocesses which share a relative part, proportional to their normalization weight, of the 1/3 * batch size.

        A minimal number of
        Due to rounding to integer number of events, the resulting batch size can differ from the desired batch size.
        To minimize this effect, the algorithm first floors the expected numbers per process ID.
        Then it distributes the remaining events according to the relative contribution of each process ID to the batch, which is calculated by the normalization weights.
        The result is again floored for process IDs with a relative contribution below the median and ceiled for process IDs with a relative contribution above the median, to minimize the difference to the desired batch size.
        Finally, if there is still a mismatch between the resulting batch size and the desired batch size, the remaining events are added to the process ID with the biggest relative contribution.

        Args:
            process_type (str): String defining the process type for which the sample size should be calculated, e.g. "hh", "tt", "dy"

        Raises:
            ValueError: The batch size is smaller than 1, which is not supported.
            TypeError: The change of type from float to int changed the value, which should not happen since the algorithm is designed to handle this case.
        """
        if self.batch_size == -1:
            raise ValueError("Batch Size < 1 is not supported. Try a number big enough to be representative")
        logger_inst.info(f"Calculate Sampler sample sizes for subprocesses of {process_type}")
        # unpack and convert to tensors for easier handling, get desired_sub_batch_size

        # -- calculate weighting for each subprocess --
        weights = []
        for pid, proc in self.process_inst[process_type].items():
            sub_process_total_normalization_weight = proc.weights_statistics["normalization_weights"]["whole_sum"]
            sub_process_ratio = self.sub_sample_ratio.get(pid, 1) # defaults to equal importance, if no pid exist
            weighted_sub_process_weight = sub_process_total_normalization_weight * sub_process_ratio
            weights.append(weighted_sub_process_weight)
        weights = torch.tensor(weights)

        min_size = torch.tensor(self.min_size)
        sub_batch_size = int(self.batch_size * self.sample_ratio[process_type])

        # start algorithm
        # 2 cases: Phase phases need to be upsampeled OR phase spaces are already reasonable high
        #
        # in 3 steps:
        # 1. downsample calculated batch size and upsample to minimal threshold
        # 2. calculate relative weight of overflow for samples above minimal threshold
        # 3. distribute overflow from 2. by ceil and floor half of the sample according to median
        # 4. reduce  all steps resulting in sub_batch_size batch composition

        # preperation: calculate batch size according to weight
        total_weight_of_process_type = weights.sum()
        ideal_sizes = sub_batch_size * weights / total_weight_of_process_type

        # add relative contribution as information to process
        for _, ds in self.process_inst[process_type].items():
            relative_weight_contribution_of_process = ds.weights_statistics["normalization_weights"]["whole_sum"] / total_weight_of_process_type
            ds.relative_weight = (
                 relative_weight_contribution_of_process * self.sample_ratio[process_type]
                )

        # step 1:
        # check if UPSAMPLING IS NECESSARY AT ALL
        floored_sizes = torch.floor(ideal_sizes) # decreases sub batch size by length of tensor
        floored_sizes = torch.maximum(floored_sizes, min_size) # increase sub batch size by different to threshold
        # calculate overflow of sub batch
        overflow_size = int(floored_sizes.sum()) - sub_batch_size

        # step 2:
        # calculate relative contribution of events above threshold

        mask_above_threshold = floored_sizes > min_size
        floored_above_threshold = floored_sizes[mask_above_threshold]
        floored_overflow_sizes = floored_above_threshold / floored_above_threshold.sum() * overflow_size
        # step 3: floor below median, ceil for weights above
        # sort to be able to use median properlly
        floored_overflow_sizes, floored_oveflow_idx = torch.sort(floored_overflow_sizes)

        # define mask for floor and ceil
        # median -> even: under median so idx len/2 - 1 (change mask to inclusive)
        # median -> uneven: exact middle idx len/2
        # even uneven median treatment
        if len(floored_oveflow_idx) % 2 == 0:
            median = floored_oveflow_idx.median()
            below_median = floored_oveflow_idx <= median
            above_median = floored_oveflow_idx > median
        else:
            median = floored_oveflow_idx.median()
            below_median = floored_oveflow_idx < median
            above_median = floored_oveflow_idx >= median

        # apply floor and ceil for targets below and above median
        floored_overflow_sizes[below_median] = torch.floor(floored_overflow_sizes[below_median])
        floored_overflow_sizes[above_median] = torch.ceil(floored_overflow_sizes[above_median])

        # step 4: apply overflow to original floored
        # combine sort indices and threshold mask
        # this returns floored sizes sorted by biggest by indices
        indices_above_threshold = torch.arange(len(floored_sizes))[mask_above_threshold][floored_oveflow_idx]
        floored_sizes[indices_above_threshold] -= floored_overflow_sizes

        # conver to int but check if values changed, just to be safe
        if (floored_sizes.to(torch.int32).sum() - floored_sizes.sum()) != 0:
            raise TypeError("Change of type changed value")
        floored_sizes = floored_sizes.to(torch.int32)

        # if one still mismatch add it on lowest or remove from highest
        very_last_remaining = int(floored_sizes.sum()) - sub_batch_size

        # TODO BETTER SCHEME
        # remove remaining from biggest sample or add more to biggest sample

        floored_sizes[indices_above_threshold[0]] -= very_last_remaining
        # if very_last_remaining <= -1:
        #     floored_sizes[indices_above_threshold[0]] -= very_last_remaining
        # elif very_last_remaining >= 1:
        #     floored_sizes[indices_above_threshold[0]] -= very_last_remaining
        # elif very_last_remaining != 0:
        #     from IPython import embed; embed(header=f"{floored_sizes.sum()} but should be {sub_batch_size}")
        #     raise ValueError(f"Resampling failed: Created batch size of size: {floored_sizes.sum()} but should be {sub_batch_size}")

        # store batch size per phase space in dataset
        logger_inst.debug({k:w.item() for k,w in zip(list(self.process_inst[process_type].keys()), floored_sizes)})

        for k, w in zip(list(self.process_inst[process_type].keys()), floored_sizes):
            self.process_inst[process_type][k].sample_size = w.item()

    def sample_batch(self, sample_from: list[str], device: torch.device=CPU_DEVICE):
        """
        This function samples a specific number of events from all datasets instances connected to the sampler.
        Which attribute is samplet is defined by a list of strings in "sample_from". By default ["continuous", "categorical", "target"] is sampled.
        In the end all so created tensors are put on *device*
        Returns a dict of tensors with keys coming from *sample_from*.

        Args:
            sample_from (list[str]): List of strings defining which attributes to sample from each dataset. Defaults to ["continuous", "categorical", "target"].
            device (torch.device, optional): _description_. Defaults to torch.device("cpu").

        Returns:
            dict[torch.Tensor]: Dict of torch tensor where each process instance contributed according to their sample_size attribute.
        """
        # sample sample_size number of events from different process instances
        events = {}
        for _, ds in sorted(self.processes().items()):
            for attribute, sampled_events in ds.sample(sample_from, device=device).items():
                if attribute not in events:
                    events[attribute] = []
                events[attribute].append(sampled_events)

        # combine to single tensors
        for attribute in events.keys():
            events[attribute] = torch.concatenate(events[attribute], dim=0).to(device)
        return events

    def set_attr(self, attribute):
        if isinstance(attribute, str):
            attribute = [attribute]

        for attr in attribute:
            if hasattr(self, attr):
                raise AttributeError(f"Attribute {attr} already exists in sampler class.")

            def _accesor(*args, dataset_attr=attr, **kwargs):
                collector = {}
                for uid, process_inst in self.processes().items():
                    process_inst_attr = getattr(process_inst, dataset_attr)
                    # when attribute is a callable function pass args and kwargs
                    if callable(process_inst_attr):
                        collector[uid] = process_inst_attr(*args, **kwargs)
                    else:
                        collector[uid] = process_inst_attr
                return collector

            setattr(self, attr, _accesor)

    def share_weights_between_sampler(self, process_sampler):
        # helper to enable ex. validation sampler to get sampling weight from training sampler
        for process_type, processes in self.process_inst.items():
            for uid, process_inst in processes.items():

                v_process_inst = process_sampler.process_inst.get(process_type, {}).get(uid)
                if v_process_inst is not None:
                    v_process_inst.relative_weight = process_inst.relative_weight


def create_sampler(
    events,
    weight_aggregator_inst,
    target_map,
    batch_size,
    min_size=1,
    train=True,
    sample_ratio=None,
    sub_sample_ratio=None,
    ):

    if sample_ratio is None:
        sample_ratio = {"dy": 0.25, "tt": 0.25, "hh": 0.5}

    # extract data from events and wrap into Datasets
    if not events:
        raise ValueError("Sampler is not created due to feeding empty events")

    # sampler has information about all processes, information required for sampling
    # and weights accross all processes
    process_sampler = ProcessSampler(
        batch_size=batch_size,
        min_size=min_size,
        sample_ratio=sample_ratio,
        target_map = target_map,
        weight_aggregator_inst=weight_aggregator_inst,
        sub_sample_ratio=sub_sample_ratio,
        )

    for uid in list(events.keys()):
        (process_type, process_id) = uid

        logger_inst.debug((
            f"Add {process_type} pid: {process_id} to " +
            ("Train" if train else "Validation") +
            " Sampler"
            ))
        arrays = events.pop(uid)

        # create target tensor from uid
        num_events = len(arrays["continuous"])
        target_value = target_map[process_type]
        target = torch.zeros(size=(num_events, len(target_map)), dtype=torch.float32)
        target[:, target_value] = 1.
        process = Process(
            continuous=arrays["continuous"],
            categorical=arrays["categorical"],
            target=target,
            normalization_weights=arrays["normalization_weights"],
            product_of_weights=arrays["product_of_weights"],
            weights_statistics=weight_aggregator_inst.weights[uid],
            evaluation_space_mask=arrays["evaluation_mask"],
            process_id=process_id,
            process_type=process_type,
            randomize=True,
        )
        process_sampler.add_process_instance(process)

    if train:
        for process_type in process_sampler.keys:
            process_sampler.calculate_sample_size(process_type=process_type)
    return process_sampler
