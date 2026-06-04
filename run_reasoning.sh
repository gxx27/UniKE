#!/bin/bash
set -u
set -o pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${HERE}"

# ---- conda --------------------------------------------------------
# We need TWO envs: `unike` for the .pkl extraction (uses old torch + the
# unified-model code) and `vllm` for the inference (vllm 0.14 + torch 2.9).
CONDA_BASE="$(conda info --base 2>/dev/null)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

KE_ENV=${KE_ENV:-unike}
VLLM_ENV=${VLLM_ENV:-vllm}

# ---- experiment matrix --------------------------------------------
ALGS=${ALGS:-"AlphaEdit MEMIT PMET"}
MODELS=${MODELS:-"ovis blip3o omnigen2"}
DATA_PATH=${DATA_PATH:-"data/UniKE.json"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
FORCE=${FORCE:-0}

# ---- vLLM knobs ---------------------------------------------------
# All work runs on these GPUs. By default we use the single tensor-parallel
# group and let vLLM saturate the cluster via continuous batching. Setting
# TP_SIZE=1 with multiple GPUS triggers data-parallel sharding (the script
# splits start_idx/end_idx evenly across GPUs and runs one vLLM engine per
# GPU in parallel).
GPUS=${GPUS:-"0,1,2,3"}
TP_SIZE=${TP_SIZE:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}

declare -A MODEL_SLUG=(
    [ovis]="AIDC-AI_Ovis-U1-3B"
    [blip3o]="BLIP3o_BLIP3o-Model-4B"
    [omnigen2]="OmniGen2_OmniGen2"
)

mkdir -p logs

latest_run_dir () {
    ls -td "results/$1/$2"/run_* 2>/dev/null | head -1
}

override_for () {
    local tag_uc=$(echo "$1" | tr '[:lower:]' '[:upper:]')
    local alg_uc=$(echo "$2" | tr '[:lower:]' '[:upper:]')
    local var="${tag_uc}_${alg_uc}_RUN_DIR"
    printf '%s' "${!var:-}"
}

NUM_GPUS=$(awk -F, '{print NF}' <<< "${GPUS}")
if (( NUM_GPUS < TP_SIZE )); then
    echo "[abort] GPUS='${GPUS}' has ${NUM_GPUS} entries but TP_SIZE=${TP_SIZE} > NUM_GPUS"
    exit 1
fi
if (( NUM_GPUS % TP_SIZE != 0 )); then
    echo "[abort] NUM_GPUS=${NUM_GPUS} is not divisible by TP_SIZE=${TP_SIZE}"
    exit 1
fi
N_DP=$(( NUM_GPUS / TP_SIZE ))

# Total prompts to generate (we get this from the dataset on-the-fly so we
# can split start_idx/end_idx across data-parallel workers).
DATASET_SIZE=$(python -c "import json; print(len(json.load(open('${DATA_PATH}'))))")
echo "[info] dataset has ${DATASET_SIZE} cases; data-parallel workers=${N_DP}; tp=${TP_SIZE}"

run_one_combo () {
    local alg=$1
    local tag=$2
    local slug="${MODEL_SLUG[$tag]:-}"
    if [[ -z "${slug}" ]]; then
        echo "[Reason][${alg}/${tag}] SKIP: unknown model tag"
        return 0
    fi

    local run_dir
    run_dir=$(override_for "${tag}" "${alg}")
    if [[ -z "${run_dir}" ]]; then
        run_dir=$(latest_run_dir "${slug}" "${alg}")
    fi
    if [[ -z "${run_dir}" ]] || [[ ! -f "${run_dir}/edited_model.pkl" ]]; then
        echo "[Reason][${alg}/${tag}] SKIP: edited_model.pkl not found (run_dir='${run_dir}')"
        return 0
    fi

    local out_dir="${run_dir}/reasoning"
    if [[ "${FORCE}" != "1" ]] && [[ -f "${out_dir}/generation_summary.json" ]]; then
        echo "[Reason][${alg}/${tag}] SKIP: ${out_dir}/generation_summary.json exists (FORCE=1 to redo)"
        return 0
    fi

    local llm_dir="${run_dir}/edited_llm_hf"
    local extract_log="logs/reasoning_vllm_${alg,,}_${tag}_extract.log"
    local infer_log_prefix="logs/reasoning_vllm_${alg,,}_${tag}"

    # --- 1) extract HF-format LLM (only if not already present) ----------
    if [[ "${FORCE}" == "1" ]] || [[ ! -f "${llm_dir}/extract_meta.json" ]]; then
        echo "[Reason][${alg}/${tag}] step 1/2: extracting LLM checkpoint -> ${llm_dir}"
        conda activate "${KE_ENV}"
        python scripts/extract_edited_llm.py \
            --edited_model_path "${run_dir}/edited_model.pkl" \
            --output_dir "${llm_dir}" \
            2>&1 | tee "${extract_log}"
        local rc_e=${PIPESTATUS[0]}
        conda deactivate
        if (( rc_e != 0 )); then
            echo "[Reason][${alg}/${tag}] FAILED at extraction (rc=${rc_e}); see ${extract_log}"
            return ${rc_e}
        fi
    else
        echo "[Reason][${alg}/${tag}] step 1/2: reusing existing ${llm_dir}/extract_meta.json"
    fi

    # --- 2) vLLM inference ---------------------------------------------
    echo "[Reason][${alg}/${tag}] step 2/2: vLLM inference (DP=${N_DP}, TP=${TP_SIZE})"
    conda activate "${VLLM_ENV}"

    if (( N_DP == 1 )); then
        # Single engine, gets the whole dataset.
        python scripts/generate_reasoning.py \
            --vllm_model_dir "${llm_dir}" \
            --model_type "${tag}" \
            --data_path "${DATA_PATH}" \
            --output_dir "${out_dir}" \
            --max_new_tokens "${MAX_NEW_TOKENS}" \
            --tensor_parallel_size "${TP_SIZE}" \
            --gpu_memory_utilization "${GPU_MEM_UTIL}" \
            --max_model_len "${MAX_MODEL_LEN}" \
            --max_num_seqs "${MAX_NUM_SEQS}" \
            --gpus "${GPUS}" \
            2>&1 | tee "${infer_log_prefix}.log"
        local rc_i=${PIPESTATUS[0]}
        conda deactivate
        if (( rc_i != 0 )); then
            echo "[Reason][${alg}/${tag}] FAILED at vLLM inference (rc=${rc_i}); see ${infer_log_prefix}.log"
            return ${rc_i}
        fi
    else
        # Data-parallel: launch one vLLM process per (TP-sized) GPU group.
        IFS=',' read -ra GPU_ARR <<< "${GPUS}"
        local pids=()
        local shard_outdir
        local fail=0
        local i
        for ((i = 0; i < N_DP; i++)); do
            local lo=$(( DATASET_SIZE * i / N_DP ))
            local hi=$(( DATASET_SIZE * (i + 1) / N_DP ))
            local gpu_subset=""
            for ((j = i * TP_SIZE; j < (i + 1) * TP_SIZE; j++)); do
                if [[ -n "${gpu_subset}" ]]; then gpu_subset+=","; fi
                gpu_subset+="${GPU_ARR[$j]}"
            done
            shard_outdir="${out_dir}/shards/dp_${i}"
            mkdir -p "${shard_outdir}"
            echo "[Reason][${alg}/${tag}]   shard ${i}: cases [${lo}, ${hi})  GPUs=${gpu_subset}"
            (
                python scripts/generate_reasoning.py \
                    --vllm_model_dir "${llm_dir}" \
                    --model_type "${tag}" \
                    --data_path "${DATA_PATH}" \
                    --output_dir "${shard_outdir}" \
                    --start_idx "${lo}" --end_idx "${hi}" \
                    --max_new_tokens "${MAX_NEW_TOKENS}" \
                    --tensor_parallel_size "${TP_SIZE}" \
                    --gpu_memory_utilization "${GPU_MEM_UTIL}" \
                    --max_model_len "${MAX_MODEL_LEN}" \
                    --max_num_seqs "${MAX_NUM_SEQS}" \
                    --gpus "${gpu_subset}"
            ) >"${infer_log_prefix}_dp${i}.log" 2>&1 &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do
            wait "${pid}" || fail=1
        done
        conda deactivate
        if (( fail != 0 )); then
            echo "[Reason][${alg}/${tag}] FAILED in one of the data-parallel workers; see ${infer_log_prefix}_dp*.log"
            return 1
        fi

        # Merge shard outputs into the canonical reasoning_<start>_<end>.json /
        # generation_summary.json files.
        echo "[Reason][${alg}/${tag}] merging ${N_DP} shards into ${out_dir}"
        python - <<PY
import json, glob, os
out_dir = "${out_dir}"
shards = sorted(glob.glob(os.path.join(out_dir, "shards", "dp_*")))
all_results = []
total_secs = 0.0
total_prompts = 0
for sd in shards:
    for f in glob.glob(os.path.join(sd, "reasoning_*.json")):
        with open(f) as fh:
            all_results.extend(json.load(fh))
    summ = os.path.join(sd, "generation_summary.json")
    if os.path.exists(summ):
        with open(summ) as fh:
            s = json.load(fh)
        total_secs = max(total_secs, s.get("generation_seconds", 0.0))
        total_prompts += s.get("num_prompts", 0)
all_results.sort(key=lambda r: r["case_id"])
n = max(r["case_id"] for r in all_results) + 1 if all_results else 0
out_file = os.path.join(out_dir, f"reasoning_0_{n}.json")
with open(out_file, "w") as fh:
    json.dump(all_results, fh, indent=2)
summary = {
    "engine": "vllm-data-parallel",
    "n_shards": len(shards),
    "num_cases": len(all_results),
    "num_prompts": total_prompts,
    "wallclock_seconds": total_secs,
    "prompts_per_sec": (total_prompts / max(total_secs, 1e-6)),
}
with open(os.path.join(out_dir, "generation_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)
print(f"[merge] wrote {out_file} ({len(all_results)} cases) and generation_summary.json")
PY
    fi

    echo "[Reason][${alg}/${tag}] done"
}

echo "==== vLLM reasoning ===="
echo "Algorithms : ${ALGS}"
echo "Models     : ${MODELS}"
echo "GPUs       : ${GPUS}  (TP=${TP_SIZE}, DP workers=${N_DP})"
echo "Data       : ${DATA_PATH}"

ANY_FAIL=0
FAILED_LIST=()
for alg in ${ALGS}; do
    for tag in ${MODELS}; do
        if ! run_one_combo "${alg}" "${tag}"; then
            ANY_FAIL=1
            FAILED_LIST+=("${alg}/${tag}")
        fi
    done
done

echo
echo "==== Summary ===="
ls -d results/*/{AlphaEdit,MEMIT,PMET}/run_*/reasoning/generation_summary.json 2>/dev/null | sort
if (( ANY_FAIL != 0 )); then
    echo "Failed combinations: ${FAILED_LIST[*]}"
    exit 1
fi
echo "All vLLM reasoning runs complete."
