export USER="hergesk"

# storage directories
export DNN_DIR=/afs/desy.de/user/h/hergesk/repos/pytorch_network_playground
export STORE_DIR=/data/dust/user/hergesk/HH_DNN # ROOT of storage
export CACHE_DIR=${STORE_DIR}/cache # directory where preprocessed data is stored as well as dataset paths
export PICTURE_DIR=${STORE_DIR}/pictures
export MODELS_DIR=${STORE_DIR}/models # saved models
export TENSORBOARD_DIR=${STORE_DIR}/tensorboard # tensorboard storage
export EVALUATION_DIR=${STORE_DIR}/evaluation # directory when evaluating outputs

# debug level: DEBUG; INFO; WARNING
export LOG_LEVEL="DEBUG"
export FILE_LOG_LEVEL="DEBUG"

# location where the input data can be found.
export ERA=prod20 # possible eras: prod14, prod20, prod24 (20 only has 22pre)
export TRAINING_ROOT="/data/dust/user/riegerma/hh2bbtautau/run3_training_data" # normal training root
# export TRAINING_ROOT="/data/dust/user/wiedersb/machine_learning_data" # quintus training root
export INPUT_DATA_DIR="${TRAINING_ROOT}/${ERA}"


# virtualenv handling
export VENV_MODE="cf" # decide which venv is used - possible values: pyenv, venv or cf

# columnflow settings, only necessary if VENV_MDOE is set to cf
export CF_ROOT="/afs/desy.de/user/h/hergesk/repos/hh2bbtautau" # your root directory of CF
export CF_USER_FLAVOR="dev" # your cf user name - used when source setup with for example 'dev'
export CF_SANDBOX="venv_hbt_dev" #  sandbox name within columnflow

# pyenv or virtualenv settings
export VENV_ROOT="/data/dust/user/${USER}/pyenv_virtualenvs" # place to look for existing virtualenvs
export PYENV_ROOT="/afs/desy.de/user/w/${USER}/.pyenv" # root of pyenv installation
export ML_ENV="ml_torch" # name of your virtualenv, so it can be activated by source setup.sh

export PATH="${PATH}:${DNN_DIR}/src" # add virtualenv to path
# flags to stop unnecessary dir checks, can be undone to recreate dirs
export SETUP_DIRS_DONE=1
