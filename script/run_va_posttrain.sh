#!/usr/bin/bash

set -euo pipefail
set -x

umask 007

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CACHE_ROOT=${CACHE_ROOT:-"${PROJECT_ROOT}/.cache"}
ASCEND_HOME=${ASCEND_HOME:-"/usr/local/Ascend"}

export PYTHONPATH="/workspace/pypi-multi-version/transformers-4.57.6:${PYTHONPATH:-}"

setup_ascend_env() {
    export LD_LIBRARY_PATH="${ASCEND_HOME}/driver/:${LD_LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="${ASCEND_HOME}/driver/lib64/driver/:${LD_LIBRARY_PATH:-}"

    local shell_opts=$-
    set +ux
    if [ -f "${ASCEND_HOME}/ascend-toolkit/set_env.sh" ]; then
        source "${ASCEND_HOME}/ascend-toolkit/set_env.sh"
    fi
    if [ -f "${ASCEND_HOME}/nnal/atb/set_env.sh" ]; then
        source "${ASCEND_HOME}/nnal/atb/set_env.sh"
    fi
    case "${shell_opts}" in
        *u*) set -u ;;
    esac
    case "${shell_opts}" in
        *x*) set -x ;;
    esac
}

if [ "${SETUP_ASCEND_ENV:-1}" != "0" ]; then
    setup_ascend_env
fi
 
NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29501"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"libero_train"} # robotwin_train, libero_train
ASCEND_TORCH_PYTHON=${ASCEND_TORCH_PYTHON:-"/opt/mamba/envs/ascend-torch/bin/python"}
PYTHON_BIN=${PYTHON_BIN:-"${ASCEND_TORCH_PYTHON}"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python interpreter '${PYTHON_BIN}' is not executable. Set PYTHON_BIN to a valid training environment." >&2
    exit 1
fi

ensure_writable_dir() {
    local var_name=$1
    local requested_dir=$2
    local fallback_dir=$3

    if [ -n "${requested_dir}" ] && mkdir -p "${requested_dir}" 2>/dev/null && [ -w "${requested_dir}" ]; then
        printf -v "${var_name}" '%s' "${requested_dir}"
    else
        mkdir -p "${fallback_dir}"
        printf -v "${var_name}" '%s' "${fallback_dir}"
    fi
}

ensure_writable_dir HF_HOME "${HF_HOME:-}" "${CACHE_ROOT}/huggingface"
export HF_HOME
ensure_writable_dir HF_DATASETS_CACHE "${HF_DATASETS_CACHE:-"${HF_HOME}/datasets"}" "${CACHE_ROOT}/huggingface/datasets"
ensure_writable_dir HF_LEROBOT_HOME "${HF_LEROBOT_HOME:-"${HF_HOME}/lerobot"}" "${CACHE_ROOT}/huggingface/lerobot"
export HF_DATASETS_CACHE
export HF_LEROBOT_HOME
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-"auto"}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-"auto"}
export WAN_VA_DISABLE_FLEX_COMPILE=${WAN_VA_DISABLE_FLEX_COMPILE:-"1"}

export WANDB_API_KEY="your key"
export WANDB_BASE_URL="your url"
export WANDB_TEAM_NAME="your team name"
export WANDB_PROJECT="your project"

## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}

## cmd setting
export TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
"${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train --config-name ${config_name} $overrides
