
#!/bin/bash
create_dirs(){
    # -----------------------------------------------------------------------------
    # Create project directory structure if not already initialized.
    #
    # Reads:
    #   CONFIG_FILE=config.sh
    #
    # Required variables from config:
    #   STORE_DIR         Parent directory
    #   CACHE_DIR         Cache storage directory
    #   PICTURE_DIR       Image output directory
    #   MODELS_DIR        Model storage directory
    #   TENSORBOARD_DIR   Tensorboard log directory
    #   SETUP_DIRS_DONE   Setup flag (0/1)
    #
    # Side effects:
    #   - Creates missing directories
    #   - Updates SETUP_DIRS_DONE in config.sh
    #
    # Returns:
    #   0 -> success / already completed
    #   1 -> directory creation failure
    # -----------------------------------------------------------------------------
    local LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CONFIG_FILE=${LOCAL_DIR}/config.sh
    source ${CONFIG_FILE}

    # just creating directories when they don't exist
    if [ "${SETUP_DIRS_DONE}" -eq 1 ]; then
        echo "Skip Directory Setup, since its already done"
        return 0
    fi

    DIRS=(
        "${CACHE_DIR}"
        "${PICTURE_DIR}"
        "${MODELS_DIR}"
        "${TENSORBOARD_DIR}"
        "${EVALUATION_DIR}"
    )

    if [ ! -d "${STORE_DIR}" ]; then
        mkdir "${STORE_DIR}" || {
            echo "[ERROR] Failed to create STORE_DIR"
            return 1
        }
    fi

    for dir in "${DIRS[@]}"; do
        if [ ! -d "${dir}" ]; then
            mkdir "${dir}" || {
            echo "[ERROR] Failed to create ${dir}"
                return 1
            }
        fi
    done

    sed -i 's/^export SETUP_DIRS_DONE=.*/export SETUP_DIRS_DONE=1/' "${CONFIG_FILE}"
    echo "[INFO] Directory setup complete"

    return 0
}

check_required_vars() {
    # simple function that checks if given variable are set
    # -n ensures that $1 is seen as arrays reference
    local -n vars_ref=$1
    local missing=()

    for var in "${vars_ref[@]}"; do
        # !VARIABLE indirect expansion
        # -> treat the content of VARIABLE as another variable and expand that
        # necessary i loop over multiple variable names, which need to be expanded

        # :-FALLBACK_VALUE if unset use FALLBACK_VALUE
        if [ -z "${!var:-}" ]; then
            missing+=("${var}")
        fi
    done


    # #array = number of elements of array
    if [ ${#missing[@]} -gt 0 ]; then
        # $array[*] expand array into string
        echo "[ERROR] Missing variables: ${missing[*]}"
        return 1
    fi
}

# venv handling
prepare_venv(){
    # check if venv root exist, return 1 if false
    _path="${VENV_ROOT}/${ML_ENV}/bin/"
    if [ ! -n "${_path}" ]; then
        echo "Venv root does not exist check ${_path}"
        return 1
    fi
}

activate_venv(){

    _path="${VENV_ROOT}/${ML_ENV}/bin"
    # activate but without prompt change
    VIRTUAL_ENV_DISABLE_PROMPT=1 source "${_path}/activate"
    finish_setup
}

# pyenv handling
prepare_pyenv(){
    if [ -n "${PYENV_ROOT}" ] && [ -n "${ML_ENV}" ]; then
        # check if pyenv exists as command, if not activate it
        # command return 0 if exist, else another number
        # >/dev etc. just silence the output
        if ! command -v pyenv >/dev/null 2>&1; then
            echo "Bootstrap Pyenv"
            export PATH="${PYENV_ROOT}/bin/:${PATH}";
            eval "$(pyenv init --path)";
            eval "$(pyenv init -)";
            eval "$(pyenv virtualenv-init -)"
        fi
    else
        echo "can' activate pyenv due too wrong PYEV_ROOT or not set ML_ENV"
        return 1
    fi
}

activate_pyenv(){
    pyenv activate "${ML_ENV}"
    finish_setup
    echo "activate virtualenv ${ML_ENV}"
}

# columnflow sandbox
prepare_cf(){
    required_vars=(
    CF_ROOT
    CF_USER_FLAVOR
    CF_SANDBOX
    )

    check_required_vars required_vars || exit 1
    cd "${CF_ROOT}"
    source setup.sh "${CF_USER_FLAVOR}"

    if [ -n "${CF_SANDBOX}" ]; then
        if ! command -v cf_sandbox >/dev/null 2>&1; then
            echo "Source your columnflow setup!"
            return 2
        fi
    fi
    return 0
    }

activate_cf(){
    echo "Activate cf sandbox at ${CF_SANDBOX}"
    finish_setup
    cd $DNN_DIR
    cf_sandbox "${CF_SANDBOX}"
}

check_if_inside_venv(){
    # check if inside virtualenv -> if inside sys prefix changes
    # Source - https://stackoverflow.com/a/15454916
    local IN_VENV=$(python -c 'import sys; print("1" if hasattr(sys, "real_prefix") else "0")')
    # -n is true if NON-ZERO, -z true if EMPTY
    if [ "${IN_VENV}" = "1" ]; then
        echo "Already inside virtualenv"
        return 4
    fi
}

setup_env() {
    # Setup Virtualenv and mount given environment in config
    check_if_inside_venv

    # prepare different setups
    if [ "${VENV_MODE}" = "pyenv" ]; then
        prepare_pyenv
    # for bachelor and master students that use columnflow sandboxes
    elif [ "${VENV_MODE}" = "cf" ]; then
        prepare_cf
    elif [ "${VENV_MODE}" = "venv" ]; then
        prepare_venv
    fi
}

activate_env(){
    if [ "${VENV_MODE}" = "pyenv" ]; then
        activate_pyenv
    # for bachelor and master students that use columnflow sandboxes
    elif [ "${VENV_MODE}" = "cf" ]; then
        activate_cf
    elif [ "${VENV_MODE}" = "venv" ]; then
        activate_venv
    fi
}

finish_setup(){
    # set flags and expand paths
    echo "Ready to go"
    export SETUP_COMPLETE=1
    # extend PS1 if environment is set properly

    # Show only the last part of the virtualenv path (env name) in the prompt
    if [[ "${VENV_MODE}" != "cf" ]]; then
        export PS1="[${VIRTUAL_ENV##*/}] ${PS1}"
    fi
    # include source of project as root for python
    export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
    return 0
    }

setup() {
    # Prevent double execution, when SETUP_COMPLETE is set (at end of script) don't run setup twice.
    if [ -n "$SETUP_COMPLETE" ]; then
        echo "Setup already completed. Skipping."
        return 0
    fi

    # create directories, when they do not exist
    create_dirs
    CREATE_DIR_COMPLETE=$?
    if [ "${CREATE_DIR_COMPLETE}" -ne 0 ]; then
        echo "Don't continue setup - creation of directories failed"
        return 1
    fi

    # Get the directory where this script is located and source config located there
    # BASH_SOURCE[0] = path of current script, get dir and resolve absolute path
    local LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$LOCAL_DIR/config.sh"

    # run env and check status if run correctly change python path and venv marker
    setup_env
    SETUP_ENV_COMPLETE=$?

    if [ "${SETUP_ENV_COMPLETE}" -eq 0 ]; then
        activate_env
        return 0
    fi
    }

setup "$@"
