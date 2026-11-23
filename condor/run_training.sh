#!/usr/bin/env bash

# Batch job created by condor/submit_training.sh, not meant to be called by hand.
# Trains with the copy of the code in the job directory, so later edits of the repository
# (e.g. of the configs) do not change queued or running jobs.
#
# Usage:
#   run_training.sh <dnn_root> <job_dir> <name> [train.py arguments ...]
#
# dnn_root: repository root, its setup.sh and config.sh provide the environment
# job_dir:  job directory containing the code copy in src/
# name:     tensorboard name of the training (-tn)
#
# REQUESTED_GPUS (set by the submit file): number of requested GPUs, the job aborts if torch sees none

dnn_root="$1"
job_dir="$2"
name="$3"
shift 3

# condor does not necessarily set USER, config.sh derives the storage paths from it
export USER="${USER:-$(id -un)}"

# the hostname of a worker node does not match naf-*, so the system is given explicitly
source "${dnn_root}/setup.sh" naf || exit 1
set -euo pipefail

# the code copy takes precedence over the editable install of the repository
export PYTHONPATH="${job_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"

# stop early if the code copy is not used or no GPU is visible although one was requested
python - "${job_dir}/src" <<'EOF'
import os
import pathlib
import sys

import torch

import bbTT.configs

code_dir = pathlib.Path(sys.argv[1]).resolve()
used_dir = pathlib.Path(bbTT.configs.__file__).resolve()
print(f"[INFO] bbTT.configs imported from {used_dir}")
if code_dir not in used_dir.parents:
    sys.exit(f"[ERROR] Expected the code copy in {code_dir}, abort")

print(f"[INFO] torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
elif int(os.environ.get("REQUESTED_GPUS", "0")) > 0:
    # without this check the training would silently fall back to the much slower CPU
    sys.exit("[ERROR] A GPU was requested, but torch does not see one, abort")
EOF

echo "[INFO] Host: $(hostname), start: $(date)"
exec python "${job_dir}/src/bbTT/train/train.py" -tn "${name}" "$@"
