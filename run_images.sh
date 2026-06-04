#!/bin/bash
set -e

# -------- GPUs (edit me) --------
GPUS=${GPUS:-"0,1,2,3"}
MODELS_PER_GPU=${MODELS_PER_GPU:-2}

# -------- Config --------
ALGS=${ALGS:-"AlphaEdit MEMIT PMET"}
MODELS=${MODELS:-"ovis blip3o omnigen2"}
DATA_PATH=${DATA_PATH:-"data/UniKE.json"}
STEPS=${STEPS:-50}
SEED=${SEED:-42}
# Toggle which modes to generate
WITH_REASONING=${WITH_REASONING:-1}
WITHOUT_REASONING=${WITHOUT_REASONING:-1}
# CPU offload is needed on 48GB GPUs; disable on 80GB.
NO_OFFLOAD_FLAG=""
if [ "${NO_OFFLOAD:-0}" = "1" ]; then
    NO_OFFLOAD_FLAG="--no_offload"
fi

declare -A MODEL_SLUG=(
    [ovis]="AIDC-AI_Ovis-U1-3B"
    [blip3o]="BLIP3o_BLIP3o-Model-4B"
    [omnigen2]="OmniGen2_OmniGen2"
)

mkdir -p logs

latest_run_dir () {
    local slug=$1
    local alg=$2
    ls -td "results/${slug}/${alg}/run_"* 2>/dev/null | head -1
}

run_images () {
    local log_tag=$1   # e.g. "AlphaEdit_ovis"
    local run_dir=$2

    if [ -z "${run_dir}" ] || [ ! -f "${run_dir}/edited_model.pkl" ]; then
        echo "[Image][${log_tag}] SKIP: edited_model.pkl not found (run_dir='${run_dir}')"
        return 0
    fi

    echo "[Image][${log_tag}] run_dir=${run_dir}  gpus=${GPUS}"

    if [ "${WITH_REASONING}" = "1" ]; then
        if [ ! -d "${run_dir}/reasoning" ]; then
            echo "[Image][${log_tag}] WARNING: reasoning dir not found; skipping post_edit mode"
        else
            echo "[Image][${log_tag}] -> post_edit (reasoning)"
            python scripts/generate_images.py \
                --mode post_edit \
                --data_path "${DATA_PATH}" \
                --edited_model_path "${run_dir}/edited_model.pkl" \
                --reasoning_dir "${run_dir}/reasoning" \
                --output_dir "${run_dir}/images_post" \
                --gpus "${GPUS}" \
                --models_per_gpu ${MODELS_PER_GPU} \
                --steps ${STEPS} --seed ${SEED} ${NO_OFFLOAD_FLAG} \
                2>&1 | tee "logs/images_${log_tag}_post.log"
        fi
    fi

    if [ "${WITHOUT_REASONING}" = "1" ]; then
        echo "[Image][${log_tag}] -> post_edit_no_reasoning (direct)"
        python scripts/generate_images.py \
            --mode post_edit_no_reasoning \
            --data_path "${DATA_PATH}" \
            --edited_model_path "${run_dir}/edited_model.pkl" \
            --output_dir "${run_dir}/images_post_no_reasoning" \
            --gpus "${GPUS}" \
            --models_per_gpu ${MODELS_PER_GPU} \
            --steps ${STEPS} --seed ${SEED} ${NO_OFFLOAD_FLAG} \
            2>&1 | tee "logs/images_${log_tag}_direct.log"
    fi

    echo "[Image][${log_tag}] done"
}

echo "==== Sequential image generation on GPUs ${GPUS} ===="
for alg in ${ALGS}; do
    for tag in ${MODELS}; do
        slug="${MODEL_SLUG[$tag]:-}"
        if [[ -z "${slug}" ]]; then
            echo "[warn] unknown model tag '${tag}', skipping"
            continue
        fi
        run_dir=$(latest_run_dir "${slug}" "${alg}")
        if [[ -z "${run_dir}" ]]; then
            echo "[Image][${alg}/${tag}] SKIP: no run_* dir under results/${slug}/${alg}"
            continue
        fi
        run_images "${alg}_${tag}" "${run_dir}"
    done
done
echo "==== Image generation complete ===="
