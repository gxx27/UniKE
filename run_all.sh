#!/bin/bash
set -u

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${HERE}"

# -------- Configuration --------
ALGS=${ALGS:-"AlphaEdit MEMIT PMET"}
MODELS=${MODELS:-"ovis blip3o omnigen2"}
GPUS_LIST=${GPUS_LIST:-"0 1 2 3 4 5 6 7"}
GPUS_COMMA=${GPUS_COMMA:-"0,1,2,3,4,5,6,7"}
DATA_PATH=${DATA_PATH:-"data/UniKE.json"}
DATASET_SIZE_LIMIT=${DATASET_SIZE_LIMIT:-2971}
NUM_EDITS=${NUM_EDITS:-100}

SKIP_EDIT=${SKIP_EDIT:-0}
SKIP_REASONING=${SKIP_REASONING:-0}
SKIP_IMAGES=${SKIP_IMAGES:-0}
SKIP_JUDGE=${SKIP_JUDGE:-0}

# Download data from HuggingFace if not present
if [[ ! -f "${DATA_PATH}" ]]; then
    echo "[Setup] Downloading UniKE dataset from HuggingFace..."
    huggingface-cli download gxx27/UniKE UniKE.json --repo-type dataset --local-dir data
fi

mkdir -p logs

echo "================================================================"
echo " UniKE end-to-end pipeline"
echo "   data       : ${DATA_PATH}"
echo "   GPUs       : ${GPUS_LIST}"
echo "   Algorithms : ${ALGS}"
echo "   Models     : ${MODELS}"
echo "================================================================"

# Phase 1: Knowledge editing
if [[ "${SKIP_EDIT}" != "1" ]]; then
    echo ""
    echo "[Phase 1/4] Knowledge editing"
    for alg in ${ALGS}; do
        ALG="${alg}" GPUS="${GPUS_LIST}" \
        DATASET_SIZE_LIMIT="${DATASET_SIZE_LIMIT}" NUM_EDITS="${NUM_EDITS}" \
            bash run_edit.sh
    done
    echo "[Phase 1] Done."
fi

# Phase 2: Reasoning generation (vLLM)
if [[ "${SKIP_REASONING}" != "1" ]]; then
    echo ""
    echo "[Phase 2/4] Reasoning generation"
    GPUS="${GPUS_COMMA}" ALGS="${ALGS}" MODELS="${MODELS}" DATA_PATH="${DATA_PATH}" \
        bash run_reasoning.sh
    echo "[Phase 2] Done."
fi

# Phase 3: Image generation
if [[ "${SKIP_IMAGES}" != "1" ]]; then
    echo ""
    echo "[Phase 3/4] Image generation"
    for alg in ${ALGS}; do
        ALG="${alg}" GPUS="${GPUS_COMMA}" DATA_PATH="${DATA_PATH}" \
            bash run_images.sh
    done
    echo "[Phase 3] Done."
fi

# Phase 4: VQA judge (OpenRouter)
if [[ "${SKIP_JUDGE}" != "1" ]]; then
    echo ""
    echo "[Phase 4/4] VQA judge (OpenRouter)"
    ALGS="${ALGS}" MODELS="${MODELS}" DATA_PATH="${DATA_PATH}" \
        bash run_judge.sh
    echo "[Phase 4] Done."
fi

echo ""
echo "================================================================"
echo " Pipeline complete."
echo "================================================================"
