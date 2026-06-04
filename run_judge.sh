#!/bin/bash
set -u
set -o pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${HERE}"

# -------- API key -----------------------------------------------------------
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
if [[ -z "${OPENROUTER_API_KEY}" ]]; then
    echo "[abort] OPENROUTER_API_KEY is not set. export OPENROUTER_API_KEY=sk-or-v1-..."
    exit 1
fi
export OPENROUTER_API_KEY

# -------- Python ------------------------------------------------------------
# Prefer the project's `ke` env if present; otherwise whatever python is on
# PATH (the script only needs httpx + pillow, both of which are tiny).
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
    if [[ -x "python" ]]; then
        PYTHON_BIN="python"
    else
        PYTHON_BIN="python"
    fi
fi
echo "[run] python: ${PYTHON_BIN}"

# -------- Knobs -------------------------------------------------------------
MODEL_ID="${MODEL_ID:-qwen/qwen3-vl-235b-a22b-instruct}"
CONCURRENCY="${CONCURRENCY:-8}"
IMAGE_MAX_DIM="${IMAGE_MAX_DIM:-512}"
JPEG_QUALITY="${JPEG_QUALITY:-88}"
MAX_TOKENS="${MAX_TOKENS:-512}"
MAX_RETRIES="${MAX_RETRIES:-4}"
BASE_BACKOFF="${BASE_BACKOFF:-2.0}"
TIMEOUT="${TIMEOUT:-180}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}"

# Optional OpenRouter rankings metadata.  Safe to leave empty.
OPENROUTER_REFERER="${OPENROUTER_REFERER:-https://github.com/AlphaEdit}"
OPENROUTER_APP_TITLE="${OPENROUTER_APP_TITLE:-AlphaEdit-KE-Judge}"
export OPENROUTER_REFERER OPENROUTER_APP_TITLE

# Experiment matrix.
ALGS="${ALGS:-AlphaEdit MEMIT PMET}"
MODELS="${MODELS:-ovis blip3o omnigen2}"
MODES="${MODES:-reasoning direct}"
DATA_PATH="${DATA_PATH:-data/UniKE.json}"

# Where to read from (read-only) and where to write to.
RESULTS_ROOT="${RESULTS_ROOT:-results}"
SUMMARY_ROOT="${SUMMARY_ROOT:-results_summary}"
IMAGES_REASONING_SUBDIR="${IMAGES_REASONING_SUBDIR:-images_post}"
IMAGES_DIRECT_SUBDIR="${IMAGES_DIRECT_SUBDIR:-images_post_no_reasoning}"
RESULTS_REASONING_SUBDIR="${RESULTS_REASONING_SUBDIR:-results}"
RESULTS_DIRECT_SUBDIR="${RESULTS_DIRECT_SUBDIR:-results_no_reasoning}"

FORCE="${FORCE:-0}"
JUDGE_LOG_TAG="${JUDGE_LOG_TAG:-}"
START_IDX="${START_IDX:-}"
END_IDX="${END_IDX:-}"

declare -A MODEL_SLUG=(
    [ovis]="AIDC-AI_Ovis-U1-3B"
    [blip3o]="BLIP3o_BLIP3o-Model-4B"
    [omnigen2]="OmniGen2_OmniGen2"
)

mkdir -p logs "${SUMMARY_ROOT}"

# Pre-flight ----------------------------------------------------------------
if [[ ! -f "${DATA_PATH}" ]]; then
    echo "[abort] DATA_PATH not found: ${DATA_PATH}"
    exit 1
fi
if [[ ! -d "${RESULTS_ROOT}" ]]; then
    echo "[abort] RESULTS_ROOT not found: ${RESULTS_ROOT}"
    exit 1
fi
${PYTHON_BIN} - <<'PY' || { echo "[abort] missing python deps (httpx, pillow)"; exit 1; }
import httpx, PIL  # noqa: F401
PY

echo "================================================================"
echo " run_judge_openrouter.sh -- VQA judge over (alg x model x mode)"
echo "   repo         : ${HERE}"
echo "   data         : ${DATA_PATH}"
echo "   results_root : ${RESULTS_ROOT} (read-only)"
echo "   summary_root : ${SUMMARY_ROOT} (writes go here)"
echo "   model_id     : ${MODEL_ID}"
echo "   concurrency  : ${CONCURRENCY}   image_max_dim=${IMAGE_MAX_DIM}"
echo "   algs         : ${ALGS}"
echo "   models       : ${MODELS}"
echo "   modes        : ${MODES}"
echo "   force=${FORCE}    checkpoint_every=${CHECKPOINT_EVERY}"
echo "================================================================"

# Helpers --------------------------------------------------------------------
latest_run_dir () {
    # Args: slug alg
    ls -td "${RESULTS_ROOT}/$1/$2"/run_* 2>/dev/null | head -1
}

vqa_count_or_zero () {
    local f="$1"
    [[ -f "${f}" ]] || { echo 0; return; }
    ${PYTHON_BIN} - "$f" <<'PY' 2>/dev/null || echo 0
import json, sys
try:
    with open(sys.argv[1]) as fp:
        data = json.load(fp)
    idx = data.get("vqa_index")
    if isinstance(idx, dict):
        good = sum(1 for v in idx.values() if isinstance(v, dict) and "error" not in v)
        print(good); raise SystemExit
    print(len(data.get("results", [])))
except SystemExit:
    raise
except Exception:
    print(0)
PY
}

# Track outcomes for the final summary.
declare -A STATUS=()
SUMMARY_KEYS=()
ANY_FAIL=0
CREDIT_STOP=0

run_one_mode () {
    # Args: alg tag run_dir summary_run_dir mode
    local alg="$1"
    local tag="$2"
    local run_dir="$3"
    local summary_run_dir="$4"
    local mode="$5"

    local image_subdir reasoning_arg out_subdir aggregate_extra
    if [[ "${mode}" == "reasoning" ]]; then
        image_subdir="${IMAGES_REASONING_SUBDIR}"
        reasoning_arg=("--reasoning_dir" "${run_dir}/reasoning")
        out_subdir="${RESULTS_REASONING_SUBDIR}"
        aggregate_extra=()
    else
        image_subdir="${IMAGES_DIRECT_SUBDIR}"
        reasoning_arg=()
        out_subdir="${RESULTS_DIRECT_SUBDIR}"
        aggregate_extra=("--no_reasoning")
    fi

    local label="${alg}/${tag}/${mode}"
    local image_dir="${run_dir}/${image_subdir}"
    local out_dir="${summary_run_dir}/${out_subdir}"
    local vqa_file="${out_dir}/vqa_results.json"
    local final_file="${out_dir}/final_results.txt"
    local log_tag_suffix="${JUDGE_LOG_TAG:+_${JUDGE_LOG_TAG}}"
    local log="logs/judge_or_${alg,,}_${tag}_${mode}${log_tag_suffix}.log"

    SUMMARY_KEYS+=("${label}")

    if [[ ! -d "${image_dir}" ]]; then
        echo "[Judge][${label}] SKIP: no images at ${image_dir}"
        STATUS["${label}"]="skip:no_images"
        return 0
    fi

    mkdir -p "${out_dir}"

    # Optional fast-skip: if the file is already complete-ish AND not forced,
    # still re-aggregate so the final txt stays in sync.
    local already
    already=$(vqa_count_or_zero "${vqa_file}")
    if [[ "${FORCE}" != "1" && "${already}" -gt 0 ]]; then
        echo "[Judge][${label}] ${vqa_file} has ${already} entries -- attempting resume (judge will skip done tasks)"
    fi

    echo "[Judge][${label}] image_dir = ${image_dir}"
    echo "[Judge][${label}] out_dir   = ${out_dir}"
    echo "[Judge][${label}] log       = ${log}"

    # If FORCE=1, drop the existing index so the evaluator restarts.
    if [[ "${FORCE}" == "1" && -f "${vqa_file}" ]]; then
        echo "[Judge][${label}] FORCE=1 -> rotating existing vqa_results.json -> .bak"
        mv "${vqa_file}" "${vqa_file}.$(date +%Y%m%d_%H%M%S).bak"
    fi

    # ---- 1) VQA via OpenRouter --------------------------------------------
    local extra_idx=()
    [[ -n "${START_IDX}" ]] && extra_idx+=("--start_idx" "${START_IDX}")
    [[ -n "${END_IDX}" ]] && extra_idx+=("--end_idx" "${END_IDX}")

    ${PYTHON_BIN} scripts/evaluate_vqa.py \
        --image_dir "${image_dir}" \
        "${reasoning_arg[@]}" \
        --data_path "${DATA_PATH}" \
        --output_file "${vqa_file}" \
        --api_key "${OPENROUTER_API_KEY}" \
        --model_id "${MODEL_ID}" \
        --concurrency "${CONCURRENCY}" \
        --checkpoint_every "${CHECKPOINT_EVERY}" \
        --max_retries "${MAX_RETRIES}" \
        --base_backoff "${BASE_BACKOFF}" \
        --timeout "${TIMEOUT}" \
        --max_tokens "${MAX_TOKENS}" \
        --image_max_dim "${IMAGE_MAX_DIM}" \
        --jpeg_quality "${JPEG_QUALITY}" \
        --log_prefix "[Judge][${label}]" \
        "${extra_idx[@]}" \
        2>&1 | tee "${log}"
    local rc_vqa="${PIPESTATUS[0]}"

    if (( rc_vqa == 42 )); then
        echo "[Judge][${label}] STOP: OpenRouter signalled insufficient credits"
        STATUS["${label}"]="credit_exhausted"
        CREDIT_STOP=1
        # Still aggregate what we have so the partial summary is up to date.
        ${PYTHON_BIN} scripts/aggregate_results.py \
            --run_dir "${run_dir}" \
            --data_path "${DATA_PATH}" \
            --output_file "${final_file}" \
            --vqa_results_file "${vqa_file}" \
            "${aggregate_extra[@]}" \
            2>&1 | tee -a "${log}" || true
        return 42
    fi

    local n_vqa
    n_vqa=$(vqa_count_or_zero "${vqa_file}")
    if (( n_vqa == 0 )); then
        echo "[Judge][${label}] FAILED: ${vqa_file} missing or empty after evaluator (rc=${rc_vqa})"
        STATUS["${label}"]="fail:empty(rc=${rc_vqa})"
        return 1
    fi

    # ---- 2) Aggregate -----------------------------------------------------
    ${PYTHON_BIN} scripts/aggregate_results.py \
        --run_dir "${run_dir}" \
        --data_path "${DATA_PATH}" \
        --output_file "${final_file}" \
        --vqa_results_file "${vqa_file}" \
        "${aggregate_extra[@]}" \
        2>&1 | tee -a "${log}"
    local rc_agg="${PIPESTATUS[0]}"
    if (( rc_agg != 0 )); then
        echo "[Judge][${label}] FAILED at aggregation (rc=${rc_agg}); see ${log}"
        STATUS["${label}"]="fail:agg_rc=${rc_agg}"
        return ${rc_agg}
    fi

    if (( rc_vqa == 0 )); then
        STATUS["${label}"]="ok:${n_vqa}"
    else
        STATUS["${label}"]="partial:${n_vqa}(rc=${rc_vqa})"
    fi
    echo "[Judge][${label}] done -> ${final_file}"
    return ${rc_vqa}
}

# Main loop ------------------------------------------------------------------
for alg in ${ALGS}; do
    for tag in ${MODELS}; do
        slug="${MODEL_SLUG[$tag]:-}"
        if [[ -z "${slug}" ]]; then
            echo "[warn] unknown model tag '${tag}', skipping"
            continue
        fi
        run_dir=$(latest_run_dir "${slug}" "${alg}")
        if [[ -z "${run_dir}" ]]; then
            echo "[Judge][${alg}/${tag}] SKIP: no run_* dir under ${RESULTS_ROOT}/${slug}/${alg}"
            SUMMARY_KEYS+=("${alg}/${tag}/*")
            STATUS["${alg}/${tag}/*"]="skip:no_run_dir"
            continue
        fi
        run_name="$(basename "${run_dir}")"
        summary_run_dir="${SUMMARY_ROOT}/${slug}/${alg}/${run_name}"
        mkdir -p "${summary_run_dir}"

        echo
        echo "----------------------------------------------------------------"
        echo "[Judge] ${alg}/${tag}  run=${run_dir}  out=${summary_run_dir}"
        echo "----------------------------------------------------------------"

        rc_combo=0
        for mode in ${MODES}; do
            run_one_mode "${alg}" "${tag}" "${run_dir}" "${summary_run_dir}" "${mode}"
            rc=$?
            if (( rc == 42 )); then
                CREDIT_STOP=1
                rc_combo=42
                break
            fi
            (( rc != 0 )) && ANY_FAIL=1
        done

        if (( CREDIT_STOP == 1 )); then
            echo "[Judge] credit exhausted -- breaking out of the sweep"
            break 2
        fi
    done
done

# Always refresh the report from whatever we have so far ---------------------
echo
echo "[report] (re)building markdown report"
${PYTHON_BIN} scripts/build_full_report.py \
    --results_root "${SUMMARY_ROOT}" \
    --algs "${ALGS}" \
    --models "ovis blip3o omnigen2" \
    --model_filter "${MODELS}" \
    --modes "${MODES}" \
    --results_reasoning_subdir "${RESULTS_REASONING_SUBDIR}" \
    --results_direct_subdir "${RESULTS_DIRECT_SUBDIR}" \
    --data_path "${DATA_PATH}" \
    --out_path "${SUMMARY_ROOT}/full_report.md" \
    2>&1 || echo "[report] WARN: report build had errors (see above)"

# Final summary --------------------------------------------------------------
echo
echo "================================================================"
echo " run_judge_openrouter.sh DONE"
echo "================================================================"
printf "%-40s  %s\n" "combo" "status"
for key in "${SUMMARY_KEYS[@]}"; do
    printf "  %-38s  %s\n" "${key}" "${STATUS[$key]:-?}"
done

# Surface combos that still have pending work --------------------------------
echo
echo "[pending] combos with failed_tasks.json (re-run when credits are topped up):"
shopt -s nullglob
any_pending=0
for f in "${SUMMARY_ROOT}"/*/*/run_*/results*/failed_tasks.json; do
    n=$(${PYTHON_BIN} -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('pending_count',0))" "$f" 2>/dev/null || echo "?")
    echo "  $f  pending=${n}"
    any_pending=1
done
if (( any_pending == 0 )); then
    echo "  (none -- everything is done)"
fi
shopt -u nullglob

if (( CREDIT_STOP == 1 )); then
    exit 42
fi
if (( ANY_FAIL != 0 )); then
    exit 2
fi
exit 0
