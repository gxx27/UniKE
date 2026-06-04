#!/usr/bin/env bash
# Create conda env `unike` (+ flash_attn) and a `vllm` env from requirements.txt.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ="${ROOT}/requirements.txt"
ENV_NAME="${CONDA_ENV_NAME:-unike}"
PY="${PYTHON_VERSION:-3.10}"

# vLLM env (used by run_reasoning.sh for fast reasoning generation).
VLLM_ENV_NAME="${VLLM_ENV_NAME:-vllm}"
VLLM_PY="${VLLM_PYTHON_VERSION:-3.10}"
SKIP_VLLM="${SKIP_VLLM:-0}"

# Official FlashAttention wheels: https://github.com/Dao-AILab/flash-attention/releases
# These manylinux wheels link against glibc >= 2.14, so they run on older hosts
# (e.g. Ubuntu 20.04 / glibc 2.31). The mjun0812 prebuilt torch2.4 wheels require
# glibc >= 2.32 and fail on such hosts with "GLIBC_2.32 not found".
FLASH_VERSION="${FLASH_VERSION:-2.7.4.post1}"
FLASH_CU_TAG="${FLASH_CU_TAG:-cu12}"        # cu11 or cu12
FLASH_TORCH_TAG="${FLASH_TORCH_TAG:-torch2.4}"
# torch 2.4 is built with the pre-C++11 ABI, so use the abiFALSE wheel.
FLASH_ABI="${FLASH_ABI:-cxx11abiFALSE}"

usage() {
  cat <<EOF
Create conda env '${ENV_NAME}' from requirements.txt and install prebuilt flash_attn.
Also create a separate '${VLLM_ENV_NAME}' env for vLLM (used by run_reasoning.sh).

Usage: $0 [--recreate]
  --recreate    Remove the env(s) first if they already exist.

Env vars: CONDA_ENV_NAME (default unike), PYTHON_VERSION (default 3.10),
          VLLM_ENV_NAME (default vllm), SKIP_VLLM=1 to skip the vLLM env,
          FLASH_VERSION (default 2.7.4.post1), FLASH_CU_TAG (cu11|cu12, default cu12),
          FLASH_TORCH_TAG (default torch2.4), FLASH_ABI (default cxx11abiFALSE).
EOF
}

RECREATE=0
for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
    --recreate) RECREATE=1 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda/Anaconda and ensure 'conda' is on PATH." >&2
  exit 1
fi

if [[ ! -f "$REQ" ]]; then
  echo "Missing ${REQ}" >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  if [[ "$RECREATE" -eq 1 ]]; then
    echo "Removing existing conda env: $ENV_NAME"
    conda env remove -n "$ENV_NAME" -y
  else
    echo "Conda env '$ENV_NAME' already exists. Re-run with --recreate to replace it." >&2
    exit 1
  fi
fi
echo "Creating conda env: $ENV_NAME (python=${PY})"
conda create -n "$ENV_NAME" "python=${PY}" -y

echo "Installing Python dependencies from requirements.txt ..."
conda run -n "$ENV_NAME" python -m pip install --upgrade pip setuptools wheel
conda run -n "$ENV_NAME" python -m pip install -r "$REQ"

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)
    FLASH_WHEEL="flash_attn-${FLASH_VERSION}+${FLASH_CU_TAG}${FLASH_TORCH_TAG}${FLASH_ABI}-cp310-cp310-linux_x86_64.whl"
    FLASH_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_VERSION}/${FLASH_WHEEL}"
    echo "Installing flash_attn from: $FLASH_URL"
    conda run -n "$ENV_NAME" python -m pip install "$FLASH_URL"
    ;;
  *)
    echo "Skipping flash_attn on $(uname -s) $(uname -m); install a matching wheel manually if needed."
    ;;
esac

echo "Verifying imports ..."
conda run -n "$ENV_NAME" python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
try:
    import flash_attn
    print("flash_attn", getattr(flash_attn, "__version__", "unknown"))
except Exception as e:
    print("flash_attn import:", e)
PY

# ---- vLLM env (run_reasoning.sh) ----------------------------------
# run_reasoning.sh uses a separate env with a recent vLLM (newer torch) for
# fast reasoning generation. Skip with SKIP_VLLM=1.
if [[ "$SKIP_VLLM" -eq 1 ]]; then
  echo "SKIP_VLLM=1 set; not creating the '${VLLM_ENV_NAME}' env."
else
  if conda env list | awk '{print $1}' | grep -qx "$VLLM_ENV_NAME"; then
    if [[ "$RECREATE" -eq 1 ]]; then
      echo "Removing existing conda env: $VLLM_ENV_NAME"
      conda env remove -n "$VLLM_ENV_NAME" -y
    else
      echo "Conda env '$VLLM_ENV_NAME' already exists; leaving it as-is (use --recreate to replace)."
    fi
  fi
  if ! conda env list | awk '{print $1}' | grep -qx "$VLLM_ENV_NAME"; then
    echo "Creating conda env: $VLLM_ENV_NAME (python=${VLLM_PY})"
    conda create -n "$VLLM_ENV_NAME" "python=${VLLM_PY}" -y
    echo "Installing vLLM ..."
    conda run -n "$VLLM_ENV_NAME" python -m pip install --upgrade pip setuptools wheel
    conda run -n "$VLLM_ENV_NAME" python -m pip install vllm
    echo "Verifying vLLM ..."
    conda run -n "$VLLM_ENV_NAME" python - <<'PY'
try:
    import vllm, torch
    print("vllm", vllm.__version__, "torch", torch.__version__)
except Exception as e:
    print("vllm import:", e)
PY
  fi
fi

echo
echo "Done. Activate the editing env with:  conda activate ${ENV_NAME}"
echo "      Reasoning (vLLM) env:           conda activate ${VLLM_ENV_NAME}"
