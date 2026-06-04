# Interpretability: Conditioning Pathway Analysis

Mechanistic analysis experiments for Section 4.3 ("The Conditioning Pathway Bottleneck") and Appendix B ("Conditioning Pathway Analysis") of the paper.

## Experiments

| Script | Description |
|--------|-------------|
| `bottleneck_analysis.py` | SVD analysis of Ovis-U1's frozen projection bottleneck; direct pathway metrics for BLIP3o/OmniGen2 |
| `conditioning_drift.py` | Multi-method comparison of L2 drift, cosine distance, and relative perturbation |
| `dit_propagation.py` | Per-block DiT sensitivity to conditioning perturbation |
| `reasoning_conditioning.py` | 4-way decomposition comparing Direct vs Reasoning-Augmented conditioning signals |

## Usage

```bash
export CUDA_VISIBLE_DEVICES=0
conda activate unike

# Run all experiments (3 models × 3 methods × 4 experiments = 36 runs)
bash interpretability/run_all.sh --num_cases 100

# Quick smoke test
bash interpretability/run_all.sh --smoke

# Run a single experiment
python interpretability/bottleneck_analysis.py \
    --model_name AIDC-AI/Ovis-U1-3B \
    --method AlphaEdit \
    --num_cases 100
```

Results are saved to `interpretability/results/`.
