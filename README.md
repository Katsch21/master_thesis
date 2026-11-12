# pytorch_network_playground

PyTorch training framework for the HH → bbττ DNN. The python package lives in `src/bbTT`.

## 1. Get the code

Login to github, click _Fork_ and then _Create fork_. You now have a personal fork at
`github.com/YOUR-USERNAME/pytorch_network_playground`.

In your terminal go to the location where you want the code to live and write:
```bash
git clone git@github.com:YOUR-USERNAME/pytorch_network_playground.git
cd pytorch_network_playground
git remote add upstream git@github.com:Bogdan-Wiederspan/pytorch_network_playground.git
```

## 2. Create your config
To setup the project run.
```bash
source setup.sh
```
The first run creates `config.sh` from `config.sh.example` and stops.

Open `config.sh` and check:

| Variable | Meaning |
| --- | --- |
| `STORE_DIR` | Root for everything the framework writes (cache, models, tensorboard, evaluation, pictures). Default: `/data/dust/user/$USER/HH_DNN` |
| `CREATE_DIR` | `1` lets `setup.sh` create the directories under `STORE_DIR` (`mkdir -p`, safe to leave on), `0` skips it |
| `TRAINING_ROOT`, `ERA` | Where the preprocessed input data lives. Ask if you do not have access. |
| `VENV_ROOT`, `ML_ENV_CHOICE` | Directory containing the virtualenvs and their name. `ML_ENV_CHOICE` contains `${ML_SYSTEM}` (naf or maxwell), so both systems get separate virtualenvs. |

## 3. Install the python environment (once per system)
NAF and Maxwell have different GPU hardware and thus can run different CUDA builds.
Each version has its own requirement file (`requirements_naf.txt`, `requirements_maxwell.txt`) and its own virtualenv (`${VENV_ROOT}/{ML_ENV_CHOICE}_<system>`).
Run the install on each system you use; it detects the system from the hostname:

```bash
# uses python3,
bash install_venv.sh
# OR when a specfic version is used
bash install_venv.sh python3.11
# OR with the system given explicitly (naf or maxwell)
bash install_env.sh python3 maxwell
```

## 4. Every new shell

```bash
source setup.sh # automatic system detection
# OR
source setup.sh maxwell # explicit system choice
# OR
source setup.sh naf
```

This activates the virtualenv of the current system, creates missing storage directories and adds the CUDA
libraries shipped with the pip packages (cudnn, nccl) to `LD_LIBRARY_PATH`, so training works even if the system
CUDA installation does not match the torch wheel.
## What you can do

The examples below show the minimal call for each script, `--help` lists all options.
Run them from `DNN_ROOT`.

### Train a model
Configure the models Hyperparameters in `src/bbTT/configs/`.
Checkpoints are written to `${MODELS_DIR}/<name>_fold<k>.pt`, logs to `TENSORBOARD_DIR`.
```bash
cd ${DNN_ROOT}
python src/bbTT/train/train.py -tn my_first_run -s -rc
```

### Train on the NAF batch system (HTCondor)
`condor/submit_training.sh` submits one training as HTCondor job, run it on a NAF login node after `source setup.sh`.
The current state of `src/` (configs and uncommitted changes included) is copied into the job directory
`${STORE_DIR}/condor/<name>` and the job trains with this copy.
Editing the configs after a submission does not change queued or running jobs,
so several trainings can be prepared one after another: adapt the configs, submit, adapt the configs, submit, ...

```bash
bash condor/submit_training.sh my_first_job                    # 1 GPU, 4 CPU, 16 GB, 24 h
bash condor/submit_training.sh --gpus 0 --cpus 8 --runtime 48 my_cpu_job
bash condor/submit_training.sh my_job -- -rc                   # arguments after -- are passed to train.py
bash condor/submit_training.sh --help                          # all options
condor_q                                                       # status of your jobs
```

Things to know:
- `<name>` names the job directory and the tensorboard run, the checkpoints are named after `TrainingConfig.save_model_name`.
  It must differ between trainings, otherwise they overwrite each other; the submission is refused if a checkpoint or another job directory already uses it.
- Jobs do not write the data cache (`-ns`), since parallel jobs would write the same files at the same time.
  Fill the cache once before, by an interactive training or one job with `--save-cache`, then submit the others.
- A job with a requested GPU aborts if torch does not see one, instead of silently training on the CPU.
- Job output is written to `job.out`/`job.err` in the job directory, `job.log` contains the HTCondor events including the used memory.
  `git_state.txt` and `git_diff.patch` record which code was used.
- Set `CONDOR_JOB_ROOT` (e.g. in `config.sh`) to put the job directories somewhere else.

### Evaluate your model
A trained model is saved in a checkpoint.
To evaluate it on either "test", "training" or "validation" set use `evaluate.py`.
Scores are saved to `${EVALUATION_DIR}/<file_path>.pt`.

```bash
python evaluation/evaluate.py --checkpoint_src "${MODELS_DIR}/my_first_run_fold0.pt" --fold 0 --evaluate_on "test" -batch_size 4096 --save_dst my_first_run
```

### Export to .pt2

Export a checkpoint to torch-export (`.pt2`) with a dynamic batch dimension.
This is necessary when running the model with "Columnflow".
A bare name is looked up in `MODELS_DIR`, but full path is also supported.

```bash
python evaluation/export/pt2.py --checkpoint_src my_first_run --fold 0
```

### Tensorboard
All plots created in the training can be looked up with "Tensorboard".
When your setup is on a remote side, you need to setup an ssh connection with the same port as Tensorboard.

So run the following code after being on a machine connected via -L option like:
```
ssh -L 18090:localhost:8080 naf-cms17
```
This reroutes on the naf the port 8080 to your Local Host port 18090.
Thus your Tensorboard would be started like this:
```bash
tensorboard --logdir "${TENSORBOARD_DIR}" --port 8080
```

Hint: You can occupy the same port multiple times.

## Debugging
`setup.sh` provides two shell functions that drop you into a debugger whenever a python script crashes with an
uncaught exception, at the line where it happened and with all local variables available:

```bash
pdb_on          # IPython shell (default), alternatives: pdb_on pdb / pdb_on ipdb
pdb_off         # back to normal tracebacks
```

Nothing in the code needs to change; the hook works through `PYTHONPATH` and `.setup/sitecustomize.py`.

## Working with git

Push your changes to your fork:

```bash
git add ...
git commit ...
git push
```

Pull in changes from upstream (then `git push` to update your fork, or "sync" the fork on github.com):

```bash
git pull upstream main
```

Open a pull request to the original repo with the _Contribute_ button on github.com.
