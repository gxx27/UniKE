#!/bin/bash
set -e

# -------- GPUs (edit me) --------
OVIS_GPU=${OVIS_GPU:-0}
BLIP3O_GPU=${BLIP3O_GPU:-1}
OMNIGEN2_GPU=${OMNIGEN2_GPU:-2}

# -------- Dataset / editing config --------
ALG=${ALG:-AlphaEdit}
DATASET_SIZE_LIMIT=${DATASET_SIZE_LIMIT:-2971}   # UniKE.json has 2971 items
NUM_EDITS=${NUM_EDITS:-100}
DS_NAME=${DS_NAME:-unike}

mkdir -p logs

run_edit () {
    local gpu=$1
    local model_name=$2
    local hparams_fname=$3
    local log_tag=$4

    local model_slug=$(echo "${model_name}" | sed 's/[\/\\]/_/g')
    echo "[Edit][${log_tag}] GPU=${gpu}  model=${model_name}"

    CUDA_VISIBLE_DEVICES=${gpu} python -m experiments.evaluate \
        --alg_name=${ALG} \
        --model_name="${model_name}" \
        --hparams_fname="${hparams_fname}" \
        --results_model_dir="${model_slug}" \
        --ds_name=${DS_NAME} \
        --dataset_size_limit=${DATASET_SIZE_LIMIT} \
        --num_edits=${NUM_EDITS} \
        --save_model \
        > "logs/edit_${log_tag}.log" 2>&1
    echo "[Edit][${log_tag}] done (log: logs/edit_${log_tag}.log)"
}

echo "==== Starting parallel editing (${ALG}) on 3 GPUs ===="
run_edit ${OVIS_GPU}     "AIDC-AI/Ovis-U1-3B"       "Ovis-U1-3B.json" "ovis"     &
PID_OVIS=$!
run_edit ${BLIP3O_GPU}   "BLIP3o/BLIP3o-Model-4B"   "BLIP3o-4B.json"  "blip3o"   &
PID_BLIP=$!
run_edit ${OMNIGEN2_GPU} "OmniGen2/OmniGen2"        "OmniGen2.json"   "omnigen2" &
PID_OMNI=$!

echo "Launched PIDs: ovis=${PID_OVIS}  blip3o=${PID_BLIP}  omnigen2=${PID_OMNI}"
echo "Tail logs with: tail -f logs/edit_{ovis,blip3o,omnigen2}.log"

FAILED=0
for pid in ${PID_OVIS} ${PID_BLIP} ${PID_OMNI}; do
    if ! wait ${pid}; then
        echo "[Edit] PID ${pid} FAILED"
        FAILED=1
    fi
done

if [ ${FAILED} -ne 0 ]; then
    echo "==== Editing finished with errors. Check logs/edit_*.log ===="
    exit 1
fi

echo "==== All three editing runs completed ===="
ls -d results/*/${ALG}/run_* 2>/dev/null | sort
