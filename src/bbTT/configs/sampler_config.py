from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal, Optional

from torch import Generator

from bbTT.configs.utils import choice_check

SAMPLING_STRATEGY = Literal["old_largest_remainder", "largest_remainder", "stochastic"]





@dataclass
class LargestRemainderRoundingConfig:
    min_events_in_batch: int = (
        1  # sampler setting: minimal number of events of a subprocess in a batch, only valid for least_remainder
    )


@dataclass
class StochasticRoundingConfig:
    generator: Optional[Generator] = None  # torch random generator, using default if None


@dataclass
class SamplerConfig:
    sampler_strategy_choice: SAMPLING_STRATEGY = "stochastic"

    sample_ratio: dict[str, float] = field(
        default_factory=lambda: {"dy": 1 / 3, "tt": 1 / 3, "hh": 1 / 3}
    )  # decide the ratio of tt, dy and hh within a batch
    sub_process_ratios: dict[str, float] = field(
        default_factory=lambda: {
            "signal": {},  # empty categorizes are set to 1 by default
            "tt": {(1100, 1200): 1, 1300: 1},  # groups are possible, and mixes are allowed
            "dy": {
                51667: 1,
                51683: 1,
                51664: 1,
                51680: 1,
                51720: 1,
                51723: 1,
                51726: 1,  # are wrong replace with new definition
                51729: 1,
                51732: 1,
                51735: 1,
                51674: 1,
                51690: 1,
                51665: 1,
                51681: 1,
                51661: 1,
                51677: 1,
                51670: 1,
                51671: 1,
                51672: 1,
                51673: 1,
                51675: 1,
                51686: 1,
                51687: 1,
                51688: 1,
                51689: 1,
                51691: 1,
                51666: 1,
                51682: 1,
                51699: 1,
                51702: 1,
                51705: 1,
                51708: 1,
                51711: 1,
                51714: 1,
                51693: 1,
                51668: 1,
                51684: 1,
                51663: 1,
                51679: 1,
            },
        }
    )  # amplify the subprocess xsec for the sampler only

    use_sub_process_ratios: tuple[str] = ("signal", "tt", "dy")  # which sub_process_ratios are chosen
    sample_attributes: tuple[str, ...] = (
        "continuous",
        "categorical",
        "targets",
        "product_of_weights",
        "evaluation_space_mask",
    )  # which attributes the sampler samples

    largest_remainder_rounding_config: LargestRemainderRoundingConfig = field(default_factory=LargestRemainderRoundingConfig) # noqa
    stochastic_rounding_config: StochasticRoundingConfig = field(default_factory=StochasticRoundingConfig)

    @property
    def active_config(self):
        return {
            "old": self.largest_remainder_rounding_config,  # old implementation just for backwards compatibility
            "largest_remainder": self.largest_remainder_rounding_config,
            "stochastic": self.stochastic_rounding_config,
        }[self.sampler_strategy_choice]

    def __post_init__(self):
        choice_check(self.sampler_strategy_choice, SAMPLING_STRATEGY)
        # calculate subprocess rates
        self.sub_process_ratios = self.multiply_sub_process_rates(self.use_sub_process_ratios, self.sub_process_ratios)

        # check for sampler
        necessary_field = ("continuous", "categorical", "targets")
        if not set(necessary_field).issubset(set(self.sample_attributes)):
            raise ValueError(f"Sample Attributes need to contain: {necessary_field}")


    @staticmethod
    def multiply_sub_process_rates(which_sub_process_group: tuple[str], sub_process_rates: dict[str, tuple[int]]):
        """
        Helper function to multiply sub process rates together up to 2 nested level down.
        Only *which_sub_process_group* specify which rates are picked from the dictionary describing *sub_process_rates* groups.

        Args:
            which_sub_process_group (tuple[str]): Iterable of sub process group names
            sub_process_rates (dict[str, tuple[int]]): Dictionary of all sub process rates, not all are used.
        """

        def validate_exactly_two_levels(d):
            for inner in d.values():
                # only dictionaries on top level are accepted
                if not isinstance(inner, dict):
                    raise TypeError("Expected dict at top-level key")

                # no more than another dict is accepted
                has_dict_values = any(isinstance(v, dict) for v in inner.values())
                if has_dict_values:
                    raise ValueError("Nesting deeper than 2 levels detected")

        validate_exactly_two_levels(sub_process_rates)
        final_rate = defaultdict(lambda: 1)
        for name in which_sub_process_group:
            for key, rate in sub_process_rates[name].items():
                # if a whole group should be handled
                if isinstance(key, Iterable) and not isinstance(key, str):
                    for sub_process in key:
                        final_rate[sub_process] *= rate
                # no nested case
                else:
                    final_rate[key] *= rate
        return dict(final_rate)
