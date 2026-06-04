#!/bin/bash
# Run all interpretability experiments across 3 models and 3 methods.
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0
#   conda activate unike
#   bash interpretability/run_all.sh [--num_cases N] [--smoke]

set -e

NUM_CASES=${NUM_CASES:-100}

while [[ $# -gt 0 ]]; do
    case $1 in
        --num_cases) NUM_CASES="$2"; shift 2 ;;
        --smoke) NUM_CASES=5; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=========================================="
echo "Interpretability Experiments"
echo "Num cases: $NUM_CASES"
echo "=========================================="

MODELS=(
    "AIDC-AI/Ovis-U1-3B"
    "BLIP3o/BLIP3o-Model-4B"
    "OmniGen2/OmniGen2"
)
METHODS=("AlphaEdit" "MEMIT" "PMET")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

mkdir -p interpretability/results

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "MODEL: $MODEL"
    echo "=========================================="

    for METHOD in "${METHODS[@]}"; do
        echo ""
        echo "--- Method: $METHOD ---"

        echo "[1/4] Bottleneck Analysis: $MODEL / $METHOD"
        python interpretability/bottleneck_analysis.py \
            --model_name "$MODEL" --method "$METHOD" --num_cases "$NUM_CASES" \
            2>&1 | tee interpretability/results/bottleneck_${MODEL//\//_}_${METHOD}.log || true

        echo "[2/4] Conditioning Drift: $MODEL / $METHOD"
        python interpretability/conditioning_drift.py \
            --model_name "$MODEL" --method "$METHOD" --num_cases "$NUM_CASES" \
            2>&1 | tee interpretability/results/drift_${MODEL//\//_}_${METHOD}.log || true

        echo "[3/4] DiT Propagation: $MODEL / $METHOD"
        python interpretability/dit_propagation.py \
            --model_name "$MODEL" --method "$METHOD" --num_cases "$NUM_CASES" \
            2>&1 | tee interpretability/results/dit_${MODEL//\//_}_${METHOD}.log || true

        echo "[4/4] Reasoning Conditioning: $MODEL / $METHOD"
        python interpretability/reasoning_conditioning.py \
            --model_name "$MODEL" --method "$METHOD" --num_cases "$NUM_CASES" \
            2>&1 | tee interpretability/results/reasoning_${MODEL//\//_}_${METHOD}.log || true

        echo "--- Done: $METHOD ---"
    done
done

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "Results in: interpretability/results/"
echo "=========================================="
