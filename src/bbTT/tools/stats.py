"""
Show the cached standardization statistics (mean, std, padding per feature) of the current config.

Usage: ``bbtt-stats --list`` (statistics files of the current cache) or ``bbtt-stats --show latest``.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib

import pandas as pd

from bbTT.tools.cli import parse_args


def statistics_files(cache_dir: pathlib.Path) -> list[pathlib.Path]:
    """
    Args:
        cache_dir (pathlib.Path): ``$CACHE_DIR/<hash>``.

    Returns:
        list[pathlib.Path]: ``training_statistics_<fingerprint>.json`` files, newest first.
    """
    return sorted(cache_dir.glob("training_statistics_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)


def fingerprint_of(path: pathlib.Path) -> str:
    """
    Args:
        path (pathlib.Path): Statistics file.

    Returns:
        str: The fingerprint hash in the file name.
    """
    return path.stem.replace("training_statistics_", "")


def overview(files: list[pathlib.Path]) -> pd.DataFrame:
    """
    One row per statistics file with what its fingerprint says.

    Args:
        files (list[pathlib.Path]): Statistics files.

    Returns:
        pd.DataFrame: Index fingerprint; columns ``eras``, ``split``, ``features``, ``dummy``, ``modified``.
    """
    rows = {}
    for path in files:
        content = json.loads(path.read_text())
        fingerprint = content["fingerprint"]
        rows[fingerprint_of(path)] = {
            "eras": " ".join(fingerprint["eras"]),
            "split": " ".join(f"{key}={value}" for key, value in fingerprint["split"].items()),
            "features": len(content["features"]),
            "dummy": fingerprint["return_dummy"],
            "modified": datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("fingerprint")


def statistics_table(path: pathlib.Path) -> pd.DataFrame:
    """
    The per-feature numbers of one statistics file.

    Args:
        path (pathlib.Path): Statistics file.

    Returns:
        pd.DataFrame: Index feature (tensor column order); columns ``mean``, ``std``, ``padding``.
    """
    content = json.loads(path.read_text())
    return pd.DataFrame(
        {"mean": content["mean"], "std": content["std"], "padding": content["padding"]},
        index=pd.Index(content["features"], name="feature"),
    )


def main(argv: list[str] | None = None) -> int:
    """
    Entry point of ``bbtt-stats``.

    Args:
        argv (list[str] | None): Command line arguments without the program name.

    Returns:
        int: Exit code, 0 on success.
    """
    parser = argparse.ArgumentParser(prog="bbtt-stats", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--list", action="store_true", help="statistics files of the current cache and their fingerprints")
    what.add_argument("--show", metavar="FINGERPRINT", help="mean/std/padding per feature of one file, 'latest' for the newest")
    parser.add_argument("--hash", help="cache hash under $CACHE_DIR, default: hash of the current DataConfig")
    args = parse_args(parser, argv)

    if args.hash is None:
        from bbTT.configs.io_config import DataConfig  # local import: keeps the bare call (help) fast

        args.hash = DataConfig().content_hash()
    cache_dir = pathlib.Path(os.environ["CACHE_DIR"]) / args.hash
    files = statistics_files(cache_dir)
    if not files:
        parser.error(f"no training_statistics_*.json in {cache_dir}")
    print(f"cache: {cache_dir}")

    with pd.option_context("display.max_rows", None, "display.width", None, "display.max_colwidth", None, "display.float_format", "{:.6g}".format):
        if args.list:
            print(overview(files).to_string())
        else:
            matches = files[:1] if args.show == "latest" else [path for path in files if fingerprint_of(path) == args.show]
            if not matches:
                parser.error(f"no statistics file with fingerprint '{args.show}', known: {', '.join(map(fingerprint_of, files))}")
            print(f"file: {matches[0].name}")
            print(statistics_table(matches[0]).to_string())
    return 0
