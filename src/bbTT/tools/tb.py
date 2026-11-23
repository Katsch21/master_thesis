"""
Start tensorboard on ``$TENSORBOARD_DIR``, optionally only for the runs whose name contains a filter.

Usage: ``bbtt-tb --all`` or ``bbtt-tb --filter monitoring_v2 --port 6007``.
"""
from __future__ import annotations

import argparse
import os
import pathlib

from bbTT.tools.cli import parse_args


def main(argv: list[str] | None = None) -> int:
    """
    Entry point of ``bbtt-tb``; replaces the process with tensorboard (``os.execvp``), so it never returns on success.

    Args:
        argv (list[str] | None): Command line arguments without the program name.

    Returns:
        int: Exit code, only reached on an error before tensorboard starts.
    """
    parser = argparse.ArgumentParser(prog="bbtt-tb", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--all", action="store_true", help="serve every run in $TENSORBOARD_DIR")
    what.add_argument("--filter", metavar="TEXT", help="serve only the runs whose directory name contains TEXT")
    parser.add_argument("--port", type=int, default=6006, help="port tensorboard listens on (default 6006)")
    parser.add_argument("--bind-all", action="store_true", help="listen on all interfaces instead of localhost only")
    args = parse_args(parser, argv)

    tensorboard_dir = pathlib.Path(os.environ["TENSORBOARD_DIR"])
    command = ["tensorboard", "--port", str(args.port)]
    if args.bind_all:
        command.append("--bind_all")

    if args.all:
        command += ["--logdir", str(tensorboard_dir)]
    else:
        runs = sorted(path for path in tensorboard_dir.iterdir() if path.is_dir() and args.filter in path.name)
        if not runs:
            parser.error(f"no run in {tensorboard_dir} contains '{args.filter}'")
        # logdir_spec keeps the run names as tensorboard would show them with --logdir
        command += ["--logdir_spec", ",".join(f"{path.name}:{path}" for path in runs)]

    print(" ".join(command))
    os.execvp(command[0], command)
    return 1  # unreachable, execvp raises on failure
