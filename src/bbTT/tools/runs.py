"""
List the trainings on disk: checkpoints in ``$MODELS_DIR`` and tensorboard runs in ``$TENSORBOARD_DIR``.

Usage: ``bbtt-runs --all`` or ``bbtt-runs --models`` / ``bbtt-runs --tensorboard`` for one of the two.
"""
from __future__ import annotations

import argparse
import datetime
import os
import pathlib
import re

import pandas as pd

from bbTT.tools.cache_ls import directory_size, human_size
from bbTT.tools.cli import parse_args

CHECKPOINT_PATTERN = re.compile(r"^(?P<name>.+)_fold(?P<fold>\d+)\.pt$")  # see checkpoint.paths.checkpoint_path


def modified(path: pathlib.Path) -> str:
    """
    Args:
        path (pathlib.Path): File or directory.

    Returns:
        str: Modification time as ``YYYY-mm-dd HH:MM``.
    """
    return datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def models_table(models_dir: pathlib.Path) -> pd.DataFrame:
    """
    One row per training name, its ``<name>_fold<k>.pt`` checkpoints grouped.

    Args:
        models_dir (pathlib.Path): ``$MODELS_DIR``.

    Returns:
        pd.DataFrame: Index name; columns ``folds``, ``size``, ``modified``; newest first.
    """
    rows: dict[str, dict] = {}
    for path in models_dir.glob("*.pt"):
        match = CHECKPOINT_PATTERN.match(path.name)
        name, fold = (match["name"], match["fold"]) if match else (path.stem, "-")
        row = rows.setdefault(name, {"folds": [], "bytes": 0, "modified": ""})
        row["folds"].append(fold)
        row["bytes"] += path.stat().st_size
        row["modified"] = max(row["modified"], modified(path))
    table = pd.DataFrame.from_dict(rows, orient="index").rename_axis("name")
    table["folds"] = table["folds"].map(lambda folds: " ".join(sorted(folds)))
    table["size"] = table.pop("bytes").map(human_size)
    return table[["folds", "size", "modified"]].sort_values("modified", ascending=False)


def tensorboard_table(tensorboard_dir: pathlib.Path) -> pd.DataFrame:
    """
    One row per tensorboard run directory ``<prefix>-<run name>`` (see TensorboardLogger.logger_path).

    Args:
        tensorboard_dir (pathlib.Path): ``$TENSORBOARD_DIR``.

    Returns:
        pd.DataFrame: Index run directory; columns ``folds``, ``size``, ``modified``; newest first.
    """
    rows = {}
    for path in tensorboard_dir.iterdir():
        if not path.is_dir():
            continue
        folds = sorted(sub.name.replace("fold", "") for sub in path.glob("fold*") if sub.is_dir())
        rows[path.name] = {"folds": " ".join(folds), "size": human_size(directory_size(path)), "modified": modified(path)}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("run").sort_values("modified", ascending=False)


def main(argv: list[str] | None = None) -> int:
    """
    Entry point of ``bbtt-runs``.

    Args:
        argv (list[str] | None): Command line arguments without the program name.

    Returns:
        int: Exit code, 0 on success.
    """
    parser = argparse.ArgumentParser(prog="bbtt-runs", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--all", action="store_true", help="checkpoints and tensorboard runs")
    what.add_argument("--models", action="store_true", help="checkpoints in $MODELS_DIR, grouped by training name")
    what.add_argument("--tensorboard", action="store_true", help="runs in $TENSORBOARD_DIR")
    args = parse_args(parser, argv)

    with pd.option_context("display.max_rows", None, "display.width", None, "display.max_colwidth", None):
        if args.all or args.models:
            models_dir = pathlib.Path(os.environ["MODELS_DIR"])
            print(f"models: {models_dir}")
            print(models_table(models_dir).to_string())
        if args.all:
            print()
        if args.all or args.tensorboard:
            tensorboard_dir = pathlib.Path(os.environ["TENSORBOARD_DIR"])
            print(f"tensorboard: {tensorboard_dir}")
            print(tensorboard_table(tensorboard_dir).to_string())
    return 0
