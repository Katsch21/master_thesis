"""
List the data caches under ``$CACHE_DIR``: hash, cached eras, size, age and which one the current config uses.

Usage: ``bbtt-cache-ls --all`` or ``bbtt-cache-ls --hash <hash>`` for the files of one cache.
"""
from __future__ import annotations

import argparse
import datetime
import os
import pathlib

import pandas as pd

from bbTT.tools.cli import parse_args
from bbTT.tools.sum_sizes import ERAS


def directory_size(path: pathlib.Path) -> int:
    """
    Sum the sizes of all files below ``path``.

    Args:
        path (pathlib.Path): Directory.

    Returns:
        int: Size in bytes.
    """
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def human_size(size: float) -> str:
    """
    Format a byte count with a fitting unit.

    Args:
        size (float): Size in bytes.

    Returns:
        str: E.g. ``"12.3 GB"`` or ``"4.0 KB"``.
    """
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1000 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1000
    return f"{size:.1f} GB"  # unreachable, keeps type checkers calm


def cache_overview(cache_root: pathlib.Path, current_hash: str) -> pd.DataFrame:
    """
    One row per cache directory, newest first.

    Args:
        cache_root (pathlib.Path): ``$CACHE_DIR``.
        current_hash (str): Hash of the current config, marked in the table.

    Returns:
        pd.DataFrame: Index hash; columns ``current``, ``eras``, ``statistics``, ``size``, ``modified``.
    """
    rows = {}
    for path in sorted(cache_root.iterdir()):
        if not path.is_dir():
            continue
        # chronological era order, unknown names (not in ERAS_CHOICE) last
        eras = sorted((file.stem for file in path.glob("*.pt")), key=lambda era: (era not in ERAS, ERAS.index(era) if era in ERAS else era))
        rows[path.name] = {
            "current": "*" if path.name == current_hash else "",
            "eras": " ".join(eras) or "-",
            "statistics": len(list(path.glob("training_statistics_*.json"))),
            "size": human_size(directory_size(path)),
            "modified": datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("hash").sort_values("modified", ascending=False)


def cache_files(cache_dir: pathlib.Path) -> pd.DataFrame:
    """
    The files of one cache with size and modification time.

    Args:
        cache_dir (pathlib.Path): ``$CACHE_DIR/<hash>``.

    Returns:
        pd.DataFrame: Index file name; columns ``size``, ``modified``.
    """
    rows = {
        file.name: {
            "size": human_size(file.stat().st_size),
            "modified": datetime.datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        }
        for file in sorted(cache_dir.iterdir())
        if file.is_file()
    }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("file")


def main(argv: list[str] | None = None) -> int:
    """
    Entry point of ``bbtt-cache-ls``.

    Args:
        argv (list[str] | None): Command line arguments without the program name.

    Returns:
        int: Exit code, 0 on success.
    """
    parser = argparse.ArgumentParser(prog="bbtt-cache-ls", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--all", action="store_true", help="overview of all caches, the current config's one marked with *")
    what.add_argument("--hash", help="files of this cache (use 'current' for the current config's hash)")
    args = parse_args(parser, argv)

    from bbTT.configs.io_config import DataConfig  # local import: keeps the bare call (help) fast

    cache_root = pathlib.Path(os.environ["CACHE_DIR"])
    current_hash = DataConfig().content_hash()

    with pd.option_context("display.max_rows", None, "display.width", None, "display.max_colwidth", None):
        if args.all:
            print(f"cache root: {cache_root}")
            print(cache_overview(cache_root, current_hash).to_string())
        else:
            cache_dir = cache_root / (current_hash if args.hash == "current" else args.hash)
            if not cache_dir.is_dir():
                parser.error(f"no cache at {cache_dir}")
            print(f"cache: {cache_dir}")
            print(cache_files(cache_dir).to_string())
    return 0
