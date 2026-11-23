"""
Show the cache hash of the current ``DataConfig`` and the entries it is computed from.

Usage: ``bbtt-config-hash --current`` or ``bbtt-config-hash --entries`` to see why the hash is what it is.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

from bbTT.tools.cli import parse_args


def main(argv: list[str] | None = None) -> int:
    """
    Entry point of ``bbtt-config-hash``.

    Args:
        argv (list[str] | None): Command line arguments without the program name.

    Returns:
        int: Exit code, 0 on success.
    """
    parser = argparse.ArgumentParser(prog="bbtt-config-hash", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--current", action="store_true", help="print the hash and whether its cache exists")
    parser.add_argument("--entries", action="store_true", help="also print the cache entries the hash is computed from (json)")
    args = parse_args(parser, argv)

    from bbTT.configs.io_config import DataConfig  # local import: keeps the bare call (help) fast

    config = DataConfig()
    cache_dir = pathlib.Path(os.environ["CACHE_DIR"]) / config.content_hash()
    state = "exists" if cache_dir.exists() else "not built"
    print(f"{config.content_hash()}  {cache_dir} ({state})")

    if args.entries:
        # cache_entries holds tuples and dicts of plain data; sort_keys matches the hashing order
        print(json.dumps(config.cache_entries(), indent=2, sort_keys=True, default=list))
    return 0
