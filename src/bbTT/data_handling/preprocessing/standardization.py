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

    def __str__(self) -> str:
        return self._str()

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
        Calculates the weighted mean and standard deviation over all processes of the sampler.

        Each process contributes its masked mean and variance. The processes are
        combined via the law of total variance, so the result describes the weighted
        mixture the network actually sees:

            Var_total = sum_i w_i * Var_i + sum_i w_i * (mu_i - mu_total)^2
                        (within processes)   (between processes)

        Padding entries are excluded via a mask, since they are placeholders and would
        otherwise pull the mean towards the dummy value. The reduction runs in float64
        because the later square root is numerically sensitive to the large dynamic
        range of the raw four-momenta.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Weighted mean and standard deviation
                per feature, shape [num_features] each (float64).

        Raises:
            ValueError: If the padding values are neither scalar nor of shape [num_features],
                or if a feature is fully masked (NaN mean) in a process.
        """
        if self.return_dummy:
            return self.dummy_statistics()

        logger_inst.info("Calculate mean and std over all subphase spaces")

        # per-process results are collected first and combined at the end,
        # since the between-term needs the total mean, which is only known afterwards
        means, variances, weights = [], [], []

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

            # masked statistics per process (float64 due to overflow issues with sqrt later)
            include_mask = ~(array == ignore_tensor)
            masked_mean = torch.masked.mean(input=array, mask=include_mask, dim=0, dtype=torch.float64)
            if torch.any(masked_mean.isnan()):
                # raise instead of embed(): embed() would hang forever in batch jobs
                raise ValueError(f"Process {uid} has a NaN mean, check for fully padded features in the sampler.")

            # a feature with 1 event returns NaN for its variance -> use variance of 1
            masked_var = torch.masked.var(input=array, mask=include_mask, dim=0, dtype=torch.float64)
            masked_var = torch.nan_to_num(masked_var, nan=1.0)

            means.append(masked_mean)
            variances.append(masked_var)
            weights.append(proc.relative_weight)

        # normalized weights as column vector [num_procs, 1] to broadcast over features
        w = torch.tensor(weights, dtype=torch.float64).unsqueeze(-1)
        w = w / w.sum()

        means = torch.stack(means, dim=0)          # [num_procs, num_features]
        variances = torch.stack(variances, dim=0)  # [num_procs, num_features]

        total_mean = (w * means).sum(dim=0)
        within = (w * variances).sum(dim=0)                       # spread inside each process
        between = (w * (means - total_mean) ** 2).sum(dim=0)      # spread of the process means
        total_std = torch.sqrt(within + between)
        return total_mean, total_std
