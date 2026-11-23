"""
Show what a checkpoint holds: training state, optimizer and scheduler state, and its config, optionally diffed.

Usage: ``bbtt-checkpoint-info my_run --fold 0`` or ``bbtt-checkpoint-info my_run --fold 0 --diff current``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
from typing import Any

import pandas as pd

from bbTT.tools.cli import parse_args

MISSING = "-"  # shown for a config field one side does not have


def load_checkpoint(name: str, fold: int) -> tuple[pathlib.Path, dict[str, Any]]:
    """
    Load the checkpoint dict of a training, data only (``weights_only=True``).

    Args:
        name (str): Training name as given to ``--save_model_name``.
        fold (int): Fold of the checkpoint.

    Returns:
        tuple[pathlib.Path, dict[str, Any]]: Resolved path and the checkpoint as written by ``CheckPoint._write``.
    """
    import torch

    from bbTT.checkpoint.paths import checkpoint_path

    path = checkpoint_path(name, fold)
    return path, torch.load(path, map_location="cpu", weights_only=True)


def stringify_keys(tree: Any) -> Any:
    """
    Recursively turn every dict key into a string, so the config can be json-dumped and flattened.
    Configs hold tuple keys, e.g. ``sub_process_ratios`` groups several process ids under one key.

    Args:
        tree (Any): Nested dicts/sequences as produced by ``dataclasses.asdict``.

    Returns:
        Any: Same structure with string keys; tensors and arrays become plain lists/scalars, other leaves unchanged.
    """
    if isinstance(tree, dict):
        return {str(key): stringify_keys(value) for key, value in tree.items()}
    if isinstance(tree, (list, tuple)):
        return [stringify_keys(value) for value in tree]
    if hasattr(tree, "tolist"):  # torch.Tensor / numpy array: comparable and printable as python values
        return tree.tolist()
    return tree


def flatten(tree: Any, prefix: str = "") -> dict[str, Any]:
    """
    Flatten nested dicts to ``a.b.c`` keys; sequences become json strings so tuples and lists compare equal.

    Args:
        tree (Any): Nested dict as produced by ``dataclasses.asdict``, keys already strings (see ``stringify_keys``).
        prefix (str): Key prefix of the current level.

    Returns:
        dict[str, Any]: Flat key -> leaf value.
    """
    if isinstance(tree, dict) and tree:  # an empty dict is a leaf, otherwise it would vanish
        flat = {}
        for key, value in tree.items():
            flat.update(flatten(value, f"{prefix}{key}."))
        return flat
    leaf = json.dumps(tree, default=str) if isinstance(tree, (list, dict)) else tree
    return {prefix.rstrip("."): leaf}


def config_diff(left: dict[str, Any], right: dict[str, Any]) -> pd.DataFrame:
    """
    Fields whose values differ between two configs, missing fields shown as ``MISSING`` instead of failing.

    Args:
        left (dict[str, Any]): Config dict, e.g. a checkpoint's ``full_config``.
        right (dict[str, Any]): Config dict to compare with.

    Returns:
        pd.DataFrame: Index field path; columns ``left``, ``right``; empty if the configs agree.
    """
    flat_left, flat_right = flatten(stringify_keys(left)), flatten(stringify_keys(right))
    rows = {
        key: {"left": flat_left.get(key, MISSING), "right": flat_right.get(key, MISSING)}
        for key in sorted(set(flat_left) | set(flat_right))
        if flat_left.get(key, MISSING) != flat_right.get(key, MISSING)
    }
    return pd.DataFrame.from_dict(rows, orient="index", columns=["left", "right"]).rename_axis("field")


def format_config(config: dict[str, Any]) -> str:
    """
    Render a full config as one block per sub config, one ``field  value`` line per setting.

    Args:
        config (dict[str, Any]): ``full_config`` as stored in the checkpoint (``dataclasses.asdict``).

    Returns:
        str: The blocks, nested settings as ``a.b`` paths, values aligned per block.
    """
    blocks = []
    for name, sub_config in stringify_keys(config).items():
        settings = flatten(sub_config) if sub_config else {}
        width = max((len(key) for key in settings), default=0)
        lines = [f"  {key:<{width}}  {value}" for key, value in settings.items()] or ["  (empty)"]
        blocks.append("\n".join([f"[{name}]", *lines]))
    return "\n\n".join(blocks)


def training_summary(path: pathlib.Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    """
    The head lines: which training state this is and which tensorboard run belongs to it.

    Args:
        path (pathlib.Path): Checkpoint file.
        checkpoint (dict[str, Any]): Loaded checkpoint.

    Returns:
        dict[str, Any]: Label -> value, in display order.
    """
    from bbTT.data_handling.utils import hash_dictionary

    config = checkpoint["full_config"]
    state = checkpoint.get("checkpoint_state_dict", {})
    return {
        "file": path,
        "fold": checkpoint["fold"],
        "iteration": checkpoint["iteration"],
        "loss (written)": checkpoint["loss"],
        "best loss (confirmed)": state.get("best_loss", MISSING),
        "no improvement count": state.get("no_improvement_count", MISSING),
        "loss_fn": config["loss_config"]["loss_fn"],
        "model": config["training_config"]["model_choice"],
        "eras": " ".join(config["dataset_config"]["eras"]),
        "k_fold": config["training_config"]["k_fold"],
        # same derivation as train.py: the run directory is "<tensorboard name>-<this hash>"
        "tensorboard run hash": hash_dictionary(config["training_config"]),
    }


def optimizer_summary(checkpoint: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Learning rate and weight decay per parameter group, and the scheduler state of both phases.

    Args:
        checkpoint (dict[str, Any]): Loaded checkpoint.

    Returns:
        tuple[pd.DataFrame, dict[str, Any]]: Param-group table (index group; lr, weight_decay, params)
        and the flattened scheduler state (tensor-free entries only).
    """
    groups = checkpoint["optimizer_state_dict"]["param_groups"]
    table = pd.DataFrame(
        {
            "lr": [group["lr"] for group in groups],
            "weight_decay": [group.get("weight_decay", MISSING) for group in groups],
            "params": [len(group["params"]) for group in groups],
        },
        index=pd.Index(range(len(groups)), name="group"),
    )
    scheduler = checkpoint.get("scheduler_state_dict") or {}
    flat = {key: value for key, value in flatten(scheduler).items() if not hasattr(value, "shape")}
    return table, flat


def main(argv: list[str] | None = None) -> int:
    """
    Entry point of ``bbtt-checkpoint-info``.

    Args:
        argv (list[str] | None): Command line arguments without the program name.

    Returns:
        int: Exit code, 0 on success.
    """
    parser = argparse.ArgumentParser(prog="bbtt-checkpoint-info", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="training name (--save_model_name), resolved via checkpoint_path in $MODELS_DIR")
    parser.add_argument("--fold", type=int, required=True, help="fold of the checkpoint")
    parser.add_argument("--optimizer", action="store_true", help="also show optimizer param groups and scheduler state")
    parser.add_argument("--config", action="store_true", help="also list the stored full_config, one block per sub config")
    parser.add_argument("--diff", metavar="OTHER", help="fields that differ from OTHER: 'current' (FullConfig()) or another checkpoint 'name[:fold]'")
    args = parse_args(parser, argv)

    path, checkpoint = load_checkpoint(args.name, args.fold)
    for label, value in training_summary(path, checkpoint).items():
        print(f"{label:<22} {value}")

    with pd.option_context("display.max_rows", None, "display.width", None, "display.max_colwidth", None):
        if args.optimizer:
            groups, scheduler = optimizer_summary(checkpoint)
            print("\noptimizer param groups:")
            print(groups.to_string())
            print("\nscheduler state:")
            for key, value in scheduler.items():
                print(f"  {key:<30} {value}")

        if args.config:
            print("\nfull_config:\n")
            print(format_config(checkpoint["full_config"]))

        if args.diff is not None:
            if args.diff == "current":
                from bbTT.configs.full_config import FullConfig

                other, other_label = dataclasses.asdict(FullConfig()), "current config"
            else:
                other_name, _, other_fold = args.diff.partition(":")
                other_path, other_checkpoint = load_checkpoint(other_name, int(other_fold) if other_fold else args.fold)
                other, other_label = other_checkpoint["full_config"], str(other_path)
            diff = config_diff(checkpoint["full_config"], other).rename(columns={"left": path.name, "right": other_label})
            print(f"\nconfig diff ({len(diff)} fields differ):")
            print(diff.to_string() if len(diff) else "  none")
    return 0
