"""
Show the feature registry: kind, dummy marker, column, era availability, and which features the current config uses.

Usage: ``bbtt-features --current`` (configured features) or ``bbtt-features --all`` (whole registry).
"""
from __future__ import annotations

import argparse

import pandas as pd

from bbTT.tools.cli import parse_args


def feature_table(names: list[str], prefix: str, used: set[str]) -> pd.DataFrame:
    """
    One row per feature of the registry.

    Args:
        names (list[str]): Feature names to list, in this order.
        prefix (str): Column prefix of the current production, see ``eras.column_prefix``.
        used (set[str]): Features the current config trains on, marked in the ``used`` column.

    Returns:
        pd.DataFrame: Index name; columns ``used``, ``kind``, ``dummy``, ``column``, ``unavailable_in``, ``values``.
    """
    from bbTT.data_handling.features import FEATURES

    rows = {}
    for name in names:
        spec = FEATURES[name]
        rows[name] = {
            "used": "*" if name in used else "",
            "kind": spec.kind,
            "dummy": "" if spec.dummy is None else spec.dummy,
            "column": spec.column_name(prefix) + (f"  <- {spec.jagged[0]}[{spec.jagged[1]}]" if spec.jagged else ""),
            "unavailable_in": " ".join(sorted(spec.unavailable_in)),
            "values": " ".join(map(str, spec.values)),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("feature")


def main(argv: list[str] | None = None) -> int:
    """
    Entry point of ``bbtt-features``.

    Args:
        argv (list[str] | None): Command line arguments without the program name.

    Returns:
        int: Exit code, 0 on success.
    """
    parser = argparse.ArgumentParser(prog="bbtt-features", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--current", action="store_true", help="the features of the current DataConfig in tensor column order")
    what.add_argument("--all", action="store_true", help="every feature of the registry, the configured ones marked with *")
    args = parse_args(parser, argv)

    from bbTT.configs.io_config import DataConfig  # local import: keeps the bare call (help) fast
    from bbTT.data_handling.features import FEATURES

    config = DataConfig()
    used = [*config.continuous_features, *config.categorical_features]
    names = list(FEATURES) if args.all else used
    print(f"eras: {' '.join(config.eras)}  production: {config.production}  configured: {len(config.continuous_features)} continuous, {len(config.categorical_features)} categorical")

    table = feature_table(names, prefix=config.data_prefix, used=set(used))
    if args.current:
        table = table.drop(columns="used")  # every listed feature is used
    with pd.option_context("display.max_rows", None, "display.width", None, "display.max_colwidth", None):
        print(table.to_string())
    return 0
