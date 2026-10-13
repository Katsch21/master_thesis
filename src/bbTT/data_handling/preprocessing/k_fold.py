from typing import Any

import torch

from bbTT.monitoring.logger.logger import get_logger

logger_inst = get_logger(__name__)


class FoldAndSplitCoordinator:
    def __init__(
        self,
        events: dict[torch.Tensor],
        c_fold: int,
        k_fold: int,
        training_percentage: float,
        seed: int = 0,
        randomize: bool = True,
    ):
        """
        Creates and manage indices for k-fold Cross-validation and splitting into training and validation data.
        *events* is a dictionary of form: (process_name, pid), event["event_id"].
        The number of folds are defined by *k_fold*, where *c_fold* is the current active test fold.
        The k-1 folds are further split into training and validation, defined by *training_percentage*.
        The indices are permutate using permutation if *randomize* is set with a given *seed*.

        Args:
            events (dict[torch.Tensor]): Dictionary with key "event_id" for each process, which are used to create the k-fold splits
            c_fold (int): Current fold. This fold is used as test fold and not included in training and validation data.
            k_fold (int): Number of folds. This defines how many splits are created, where one is used as test fold and the rest is split into training and validation data.
            training_percentage (float): Percentage of training data in the split of the k-1 folds. Needs to be in range of (inclusive) 0 and 1.
            seed (int, optional): Seed for the randomizer of the indices. Defaults to 0.
            randomize (bool, optional): If True randomize order of events after split. Defaults to True.

        Raises:
            ValueError: If *k_fold* is smaller or equal to 0, since at least one fold is necessary for a test split.
            ValueError: If *training_percentage* is not in range of (inclusive) 0 and 1, since this defines the split of training and validation data.
        """

        if k_fold <= 0:
            raise ValueError("k_fold parameter needs to be > 0")

        if (training_percentage > 1) or (training_percentage < 0):
            raise ValueError(
                "Training percentage needs to be in range of (inclusive) 0 and 1"
            )

        self.current_fold = c_fold
        self.k_fold = k_fold
        self.seed = seed
        self.percentage_training = training_percentage
        self.randomize = randomize

        self.indices = self.split_index_to_sets(events=events)

    def fingerprint(self) -> dict[str, Any]:
        """
        Unique information that describes WHICH events end up in the training set.
        Equal fingerprints => same training events and ordering.

        Returns:
            dict[str, Any]: JSON-serializable split parameters.
        """
        return {
            "current_fold": self.current_fold,
            "k_fold": self.k_fold,
            "seed": self.seed,
            "training_percentage": self.percentage_training,
            "randomize": self.randomize,
        }

    def split_index_to_sets(self, events):
        # TODO Docstring
        indicies = {}
        for (process_name, pid), value in events.items():
            test_fold, training_fold = self.create_fold_index_map(value["event_id"])

            # further split into train and validation
            training_id, validation_id = self.split_array_to_train_variation_by_ratio(
                training_fold
            )
            indicies[(process_name, pid)] = {
                "test": test_fold,
                "training": training_id,
                "validation": validation_id,
            }
        return indicies

    def seed_per_process(self, uid):
        # returns a seed that is unique per uid.
        # only interesting for permutation related random generators
        # reason is that permutation with same generator and same length ,results in same permutation.
        # thus can correlated these process.... is very likely a small effect.
        return self.seed + hash(uid) % (2**31 - 1)

    def create_fold_index_map(self, event_id: torch.Tensor) -> tuple[torch.Tensor]:
        """
        Creates a map of indices for test and training fold from *event_id* using *k_fold* and *current_fold*.
        The indices are permutate using *seed* if *randomize* is set to True.

        Args:
            event_id (torch.tensor): Tensor of event ids that are unique for each event

        Returns:
            tuple[torch.Tensor]: Tensor of test and training indices
        """
        test_fold_mask = event_id % self.k_fold == self.current_fold
        indices = torch.arange(len(event_id))

        # apply mask and randomize
        # ATTENTION: it is not the same to randomize the indices and the mask and then do the cut
        test_id = indices[test_fold_mask]
        training_id = indices[~test_fold_mask]
        if self.randomize:
            test_id = test_id[
                torch.randperm(
                    len(test_id), generator=torch.Generator().manual_seed(self.seed)
                )
            ]
            training_id = training_id[
                torch.randperm(
                    len(training_id), generator=torch.Generator().manual_seed(self.seed)
                )
            ]
        return test_id, training_id

    def split_array_to_train_variation_by_ratio(
        self, array: torch.Tensor
    ) -> tuple[torch.Tensor]:
        """
        Splits given *array* into training and validation parts according to the *percentage_training*.

        Args:
            array (torch.Tensor, numpy.Array): flat torch or numpy array

        Returns:
            tuple (torch.Tensor, numpy.Array): Tuple of trainings and validation array
        """
        train_length = int(round((len(array) * self.percentage_training)))
        t_idx = array[:train_length]
        v_idx = array[train_length:]
        return t_idx, v_idx

    def apply_indices(
        self,
        events: torch.Tensor,
        kind: str = "training",
        columns: tuple[str] = (
            "continuous",
            "categorical",
            "event_id",
            "normalization_weights",
            "product_of_weights",
            "evaluation_mask",
        ),
    ) -> dict[torch.Tensor]:
        """
        Apply indices of specific *kind* of specific *columns* of *events*.

        Args:
            events (dict[torch.Tensor]): Dictionary of processes
            which (str, optional): Description which kind of data is processed. Defaults to "training".
            columns (tuple[str], optional): Iterable of column names on which indices are applied to. Defaults to ("continuous", "categorical", "event_id", "normalization_weights", "product_of_weights", "evaluation_mask").

        Raises:
            ValueError: Raises error if *which* is not one of "training", "validation" or "test", since these are the only valid keys for the indices.

        Returns:
            dict[torch.Tensor]: Dictionary of events with applied indices of specific *kind* on specific *columns*.
        """
        self.invalid_kinds(kind)
        splitted_events = {}
        for uid, arrays in events.items():
            arrays = events[uid]
            splitted_events[uid] = {}
            for key in columns:
                array = arrays[key]

                splitted_array = array[self.indices[uid][kind]]
                splitted_events[uid][key] = splitted_array
        # edge case split results in empty tensors (due to very low event count).
        # just stop applying indices and remove this process from the splitted_events
        for uid in list(splitted_events.keys()):
            if splitted_events[uid]["continuous"].numel() == 0:
                logger_inst.warning(
                    f"removed {uid} from splitted events since zero elements left after k-fold split"
                )
                splitted_events.pop(uid)
        return splitted_events

    def invalid_kinds(self, kinds):
        if isinstance(kinds, str):
            kinds = [kinds]
        invalid_kinds = set(kinds) - {"training", "validation", "test"}
        if invalid_kinds:
            raise ValueError(
                f"kinds must be a subset of training, validation, test, got invalid entries: {invalid_kinds}"
            )

    def split_and_free(
        self,
        events: dict[tuple[str, int], dict[str, torch.Tensor]],
        kinds: tuple[str, ...] = ("training", "validation"),
        columns: tuple[str, ...] = (
            "continuous",
            "categorical",
            "event_id",
            "normalization_weights",
            "product_of_weights",
            "evaluation_mask",
        ),
    ) -> tuple[
        dict[tuple[str, int], dict[str, torch.Tensor]],
        dict[tuple[str, int], dict[str, torch.Tensor]],
    ]:
        """
        Splits *events* into *kinds* sets while releasing memory as it goes.
        This function depopulates inplace events, thus first getting for example training and then validation
        is not possible, both needs to be getting at the same time.

        Difference to apply_indices: Apply_indices doesn't touch *events* at all.

        Note:
            Mutates *events* in place (pops each column as it's consumed).
            Call anything that still needs full *events* (e.g. WeightAggregator)

        Args:
            events (dict[tuple[str,str]]): Dictionary of processes, each holding a dict of column tensors.
            kinds (tuple[str, ...]): Which datasets are packed together.
            columns (tuple[str, ...]) : Column names to split.

        Returns:
            Tuple of (train_events, validation_events), same nested structure as *events*.
        """
        self.invalid_kinds(kinds)

        split_events: dict[str, dict[tuple[str, int], dict[str, torch.Tensor]]] = {
            kind: {} for kind in kinds
        }

        for uid, arrays in events.items():
            for kind in kinds:
                split_events[kind][uid] = {}
            for key in columns:
                array = arrays.pop(
                    key
                )  # remove from source now: freed once every requested slice below is made
                for kind in kinds:
                    split_events[kind][uid][key] = array[self.indices[uid][kind]]
                del (
                    array
                )  # drop the last reference explicitly before moving to the next column

        # edge case: same zero-event handling as apply_indices, applied independently per kind
        for kind in kinds:
            for uid in list(split_events[kind].keys()):
                if split_events[kind][uid]["continuous"].numel() == 0:
                    logger_inst.warning(
                        f"removed {uid} from splitted events since zero elements left after k-fold split"
                    )
                    split_events[kind].pop(uid)

        return split_events

    def __call__(
        self,
        events,
        which="training",
        columns=(
            "continuous",
            "categorical",
            "event_id",
            "normalization_weights",
            "product_of_weights",
            "evaluation_mask",
        ),
    ):
        return self.apply_indices(events, which, columns)
