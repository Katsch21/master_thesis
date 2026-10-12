from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch

from bbTT.data_handling.cache import BaseCacher
from bbTT.monitoring.logger.logger import get_logger

logger_inst = get_logger(__name__)


if TYPE_CHECKING:
    from bbTT.configs.io_config import DataConfig
    from bbTT.data_handling.sampling.sampler import ProcessSampler


class FeatureStatisticCache(BaseCacher):
    def __init__(
        self, dataset_config: DataConfig, sampler: ProcessSampler, return_dummy: bool = False, verbose: bool = True
    ):
        """
        Provides the per-feature normalization statistics, cached on disk.

        On construction the statistics are either loaded from the cache file belonging
        to the config hash, or computed from the sampler and written there. The cache
        path is ``$CACHE_DIR/<content_hash>/training_statistics.json``, so a change of
        the dataset definition automatically maps to a different file instead of
        silently reusing stale values.

        Args:
            dataset_config (DataConfig): Config providing ``content_hash()``, ``continuous_features`` and ``dummy_values``.
            sampler (ProcessSampler): Sampler holding the registered processes.
            return_dummy (bool): Skip the computation and use an identity transform (mean 0, std 1). Useful when pretrained weights are loaded anyway.
            verbose (bool): Log the resulting per-feature statistics.

        Raises:
            FileExistsError: If the cache root directory does not exist.
        """
        super().__init__(config=dataset_config)
        self.sampler = sampler

        self.path = self.cache_root / self.hash / "training_statistics.json"

        self.return_dummy = return_dummy

        self.features = self.config.continuous_features
        self.padding_dummy = self.config.dummy_values

        self.mean, self.std = None, None
        if not self.path.exists():
            self.mean, self.std = self.get_statistics()
            self.save()
        else:
            self.mean, self.std = self.load()

        if verbose:
            self.verbose_statistics()

    def load(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reads mean and std from the cache file.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Mean and std as float32 tensors.
        """
        content = json.loads(self.path.read_text())
        mean = torch.tensor(content["mean"], dtype=torch.float32)
        std = torch.tensor(content["std"], dtype=torch.float32)
        logger_inst.info(f"Loaded mean and std from file {self.path}")
        return mean, std

    def save(self):
        """
        Writes the current statistics to the cache file.

        The features and padding values are stored alongside the numbers so the
        file stays self-describing and can be inspected without the config.
        """

        content = {
            "mean": self.mean.numpy().tolist(),
            "std": self.std.numpy().tolist(),
            "padding": self.padding_dummy,
            "features": self.features,
        }

        tmp_path = self.path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(content, indent=2))
        tmp_path.replace(self.path)
        logger_inst.info(f"Saved normalization statistics to {self.path}")

    def get_statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatches to the dummy or the real computation.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Mean and std per feature.
        """
        if self.return_dummy:
            mean, std = self.dummy_statistics()
        else:
            mean, std = self.compute_statistics()
        return mean, std

    def dummy_statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Builds statistics describing an identity transformation.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Zeros and ones, shape [num_features].
        """
        num_f = len(self.features)
        mean = torch.zeros(num_f)
        std = torch.ones(num_f)
        logger_inst.info(
            "\nNo normalization statistics are calculated, since return_dummy is True."
            "Returning dummy values: mean = 0 and std = 1"
        )
        return mean, std

    def __str__(self):
        self._str()

    def verbose_statistics(self):
        """
        Logs one line per feature with its mean and width at debug level.
        """

        msg = []
        for f_name, f_mean, f_var in zip(self.features, self.mean, self.std):
            msg.append(f"{f_name:<30}: mean:{f_mean:>10.4} var:{f_var:>10.4}")
        msg = "\n" + "\n".join(msg)
        logger_inst.debug(msg)

    def compute_statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the weighted mean and standard deviation over all subphase spaces of the sampler.

        Each registered process contributes its masked mean and variance, weighted
        by its relative weight. Padding entries are excluded via a mask, since they
        are placeholders and would otherwise pull the mean towards the dummy value.
        The reduction runs in float64 because the later square root is numerically
        sensitive to the large dynamic range of the raw four-momenta.

        Args:
            sampler (ProcessSampler): Sampler managing the processes to compute statistics over.
            padding_values (int, optional): List of padding_values per feature, or single value. Padding value are ignored in the calculation of the statistics. Defaults to None, which means no padding.
            return_dummy (bool): Return dummy values that describe Identity transformation. This does not compute the statistics, and are a good option when one load pretrained weights anyway.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Weighted mean and weighted standard deviation per feature, shape [num_features] each.


        Raises:
            ValueError: If the padding values are neither scalar nor of shape [num_features].

        """
        if self.return_dummy:
            return self.dummy_statistics()

        logger_inst.info("Calculate mean and std over all subphase spaces")
        weighted_means = []
        weighted_vars = []

        # uid-keyed throughout — no more relying on two separately-generated dicts
        # happening to share the same key space (that was pid-only, coincidental,
        # and no longer holds now that set_attr's accessors are gone).

        # get all processes that are registered
        processes = self.sampler.registry.all()  # dict[uid, Process]

        for current_pid_idx, (uid, proc) in enumerate(processes.items(), start=1):
            array = proc.continuous  # [num_events, num_features]

            logger_inst.info_progress(f"\rcalculating stats for pids: {current_pid_idx}/{len(processes)}")

            num_f = array.shape[-1]

            ignore_tensor = torch.tensor(self.padding_dummy)
            if ignore_tensor.shape == torch.Size([]):
                ignore_tensor = torch.full((1, num_f), fill_value=self.padding_dummy)
            elif ignore_tensor.shape != torch.Size([num_f]):
                raise ValueError(
                    f"Padding values need to be of shape [num_features] or single value, got {ignore_tensor.shape}"
                )

            # calculate masked statistics
            # doing this calculation with float 64 due to overflow issues when using sqrt later
            include_mask = ~(array == ignore_tensor)
            masked_mean = torch.masked.mean(input=array, mask=include_mask, dim=0, dtype=torch.float64)
            if torch.any(masked_mean.isnan()):
                from IPython import embed
                embed(header=f"{uid} is nan check feature_array and sampler")

            # a feature with 1 event will return a NaN for its variance.
            # in this case a variance of 1 should be used
            masked_var = torch.masked.var(input=array, mask=include_mask, dim=0, dtype=torch.float64)
            masked_var = torch.nan_to_num(masked_var, nan=1.0)

            weighted_means.append(masked_mean * proc.relative_weight)
            weighted_vars.append(masked_var * proc.relative_weight)

        sum_of_weights = sum(proc.relative_weight for proc in processes.values())
        w_avg_mean = torch.sum(torch.stack(weighted_means, axis=0), axis=0) / sum_of_weights
        w_avg_var = torch.sum(torch.stack(weighted_vars, axis=0), axis=0) / sum_of_weights
        w_avg_std = torch.sqrt(w_avg_var)
        return w_avg_mean, w_avg_std
