import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import hashlib
import json
import shutil
from itertools import islice
from time import time
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import UniKEDataset
from experiments.py.eval_utils_unike import compute_rewrite_quality_unike
from memit import MEMITHyperParams
from memit.compute_z import get_module_input_output_at_words, compute_z
from memit.memit_main import apply_memit_to_model, get_context_templates
from AlphaEdit import AlphaEditHyperParams
from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_cov
from util import nethook
from util.globals import *
from pmet import PMETHyperParams
from pmet.pmet_main import apply_pmet_to_model
from util.model_utils import load_model_and_tok, get_model_type, is_shared_visual_model

ALG_DICT = {
    "AlphaEdit": (AlphaEditHyperParams, apply_AlphaEdit_to_model),
    "PMET": (PMETHyperParams, apply_pmet_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
}

DS_DICT = {
    "unike": (UniKEDataset, compute_rewrite_quality_unike),
}


def _sanitize_model_name_for_results(
    model_name: Union[str, Tuple], hparams_fname: str
) -> str:
    """Top-level results folder from MODEL_NAME (HF ids use org/name → org_name)."""
    if not isinstance(model_name, str):
        return Path(hparams_fname).stem
    slug = model_name.replace("\\", "_").replace("/", "_").strip()
    return slug if slug else Path(hparams_fname).stem


def _alphaedit_null_space_project_path(
    model_name_str: str,
    hparams: AlphaEditHyperParams,
    results_model_dir: Optional[str],
) -> Path:
    """Cache null_space_project.pt per (model, hparams): P depends on layers and weight shapes."""
    slug = (
        results_model_dir
        if results_model_dir is not None
        else _sanitize_model_name_for_results(model_name_str, hparams.model_name)
    )
    projector_cfg = {
        "layers": hparams.layers,
        "rewrite_module_tmp": hparams.rewrite_module_tmp,
        "mom2_dataset": hparams.mom2_dataset,
        "mom2_n_samples": hparams.mom2_n_samples,
        "mom2_dtype": hparams.mom2_dtype,
        "nullspace_threshold": hparams.nullspace_threshold,
        "nullspace_cov_reg": getattr(hparams, "nullspace_cov_reg", 0.0),
    }
    cache_key = hashlib.md5(
        json.dumps(projector_cfg, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    path = Path("null_space_cache") / slug / f"null_space_project_{cache_key}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: Union[str, Path, None] = None,
    results_model_dir: Optional[str] = None,
    num_edits: int = 1,
    use_cache: bool = False,
    save_model: bool = False,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Under RESULTS_DIR: custom dir_name (e.g. sweeps) or <model_slug>/<alg_name>
    if dir_name is not None:
        rel_results = Path(dir_name)
    else:
        top = (
            results_model_dir
            if results_model_dir is not None
            else _sanitize_model_name_for_results(model_name, hparams_fname)
        )
        rel_results = Path(top) / alg_name

    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    if (
        continue_from_run is None
        or not (run_dir := RESULTS_DIR / rel_results / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / rel_results
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / rel_results / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")
    if "MEMIT" in alg_name:
    # Get run hyperparameters
        params_path = (
            run_dir / "params.json"
            if continue_from_run is not None
            else HPARAMS_DIR / "MEMIT" / hparams_fname
        )
    else:
        params_path = (
            run_dir / "params.json"
            if continue_from_run is not None
            else HPARAMS_DIR / alg_name / hparams_fname
        )
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    if type(model_name) is str:
        print(f"Instantiating model: {model_name}")
        model, tok = load_model_and_tok(model_name, device="cuda")
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    # Load data
    print("Loading dataset")
    snips = None
    vec = None

    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)
    # Get cache templates
    cache_template = None
    if use_cache:
        if any(alg in alg_name for alg in ["MEMIT","AlphaEdit", "MEMIT_seq", "MEMIT_prune", "MEMIT_rect"]):
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_MEMIT"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
            )
        else:
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
            )
        print(f"Will load cache from {cache_template}")
    if alg_name == "NSE":
        cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        for record in ds:
            # Retrieve k/v pair if already stored in cache
            cache_fname = (
                Path(
                    str(cache_template).format(
                        hparams.layers[-1], hparams.clamp_norm_factor, record["case_id"]
                    )
                )
                if cache_template is not None
                else None
            )
            data_loaded = False
            if (
                cache_fname is not None  # Require cache template
                and cache_fname.exists()  # Cache file must exist
            ):
                continue
            # Compute k/v pair if not loaded from cache
            if not data_loaded:
                context_templates = get_context_templates(model, tok)
                cur_z = compute_z(
                    model,
                    tok,
                    {"case_id": record["case_id"], **record["requested_rewrite"]},
                    hparams,
                    hparams.layers[-1],
                    context_templates,
                )
                if cache_fname is not None:
                    cache_fname.parent.mkdir(exist_ok=True, parents=True)
                    np.savez(
                        cache_fname,
                        **{
                            "v_star": cur_z.detach().cpu().numpy(),
                        },
                    )
                    print(f"Cached k/v pair at {cache_fname}")
    if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "MEMIT_prune", "NSE", "PMET"]):
        # Iterate through dataset
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        if hparams.model_name == "gpt2-xl":
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
            if alg_name == "AlphaEdit":
                P = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
        else:
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
            if alg_name == "AlphaEdit":
                P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
        del W_out
    if alg_name == "AlphaEdit":
        p_path = _alphaedit_null_space_project_path(
            model_name, hparams, results_model_dir
        )
        if p_path.exists():
            print(f"Loading null-space projector P from {p_path}")
            P = torch.load(p_path)
        else:
            print(f"Computing null-space projector P (will cache to {p_path})")
            for i, layer in enumerate(hparams.layers):
                P[i, :, :] = get_project(model, tok, layer, hparams)
            torch.save(P, p_path)
            print(f"Saved null-space projector P to {p_path}")
    # hs = get_module_input_output_at_words(
    #         model,
    #         tok,
    #         hparams.layers[-1],
    #         context_templates=[request["template"] for request in eval_ds],
    #         words=[request["subject"] for request in eval_ds],
    #         module_template=hparams.layer_module_tmp,
    #         fact_token_strategy=hparams.fact_token,
    #     )[1].T
    # torch.save(hs, "pre_edit_hs.pt")
    # del hs
    cnt = 0
    # Create edit_case folder for case result files
    edit_case_dir = run_dir / "edit_case"
    edit_case_dir.mkdir(parents=True, exist_ok=True)
    
    for record_chunks in chunks(ds, num_edits):
        case_result_template = str(edit_case_dir / "{}_edits-case_{}.json")
        print(f"=================================================================={cnt+1}_edit==================================================================")
        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(num_edits, record["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            continue
        
        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT","AlphaEdit", "MEMIT_seq", "MEMIT_prune", "NSE", "PMET"]) else dict()
        seq_args = dict(cache_c=cache_c) if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "NSE", "PMET"]) else dict()
        nc_args = dict(P = P) if any(alg in alg_name for alg in ["AlphaEdit"]) else dict()
        # if cnt == 0 and args.downstream_eval_steps > 0:#do initial GLUE EVAL WITH ORIGINAL MODEL
        #     glue_results = {'edit_num': -1}

        #     out_file = glue_save_location + "base.json"
            
        #     glue_eval = GLUEEval(model, tok, number_of_tests = 100)
        #     glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)

        #     #store the individual overall result file
        #     output_filename = out_file.replace('.json', '_glue.json')
        #     with open(output_filename, "w") as f:
        #         json.dump(glue_results, f, indent=4)
        start = time()
        if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "NSE", "PMET"]):
            edited_model, cache_c = apply_algo(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **rewrite_dict}
                    for record in record_chunks
                    for rewrite_dict in (
                        record["requested_rewrite"]
                        if isinstance(record["requested_rewrite"], list)
                        else [record["requested_rewrite"]]
                    )
                ],
                hparams,
                **args_conserve_memory,
                **etc_args,
                **seq_args,
                **nc_args,
            )
        elif alg_name == "MEMIT_prune":
            if cnt == 0:
                edited_model, weights_copy = apply_algo(
                    model,
                    tok,
                    [
                        {"case_id": record["case_id"], **rewrite_dict}
                        for record in record_chunks
                        for rewrite_dict in (
                            record["requested_rewrite"]
                            if isinstance(record["requested_rewrite"], list)
                            else [record["requested_rewrite"]]
                        )
                    ],
                    hparams,
                    return_orig_weights=True,
                    **args_conserve_memory,
                    **etc_args,
                )
                # Initialize the upd_matrix dictionary
                upd_matrix = {}
            else:
                edited_model, _ = apply_algo(
                    model,
                    tok,
                    [
                        {"case_id": record["case_id"], **rewrite_dict}
                        for record in record_chunks
                        for rewrite_dict in (
                            record["requested_rewrite"]
                            if isinstance(record["requested_rewrite"], list)
                            else [record["requested_rewrite"]]
                        )
                    ],
                    hparams,
                    return_orig_weights=False,
                    **args_conserve_memory,
                    **etc_args,
                )
            if cnt == (dataset_size_limit/num_edits) - 1:
            # Calculate the weight update matrix
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        current_weight = nethook.get_parameter(model, k)
                        upd_matrix[k] = current_weight - v.to("cuda")
                        # Calculate max singular value of the original weight
                        _, S_orig, _ = torch.svd(v)
                        max_sigma = S_orig.max().item()

                        # Adjust the upd_matrix singular values
                        U_upd, S_upd, V_upd = torch.svd(upd_matrix[k])
                        adjusted_S = torch.where(
                            S_upd > max_sigma,
                            torch.log(S_upd) - torch.log(torch.tensor(max_sigma, device='cuda')) + max_sigma,
                            S_upd
                        )
                        upd_matrix[k] = torch.matmul(U_upd, torch.matmul(torch.diag(adjusted_S), V_upd.t()))

                # Apply the adjusted updates to the model
                with torch.no_grad():
                    for k in upd_matrix:
                        original_weight = nethook.get_parameter(model, k)
                        adjusted_weight = original_weight + upd_matrix[k]
                        original_weight.copy_(adjusted_weight)
        else:
            edited_model, _ = apply_algo(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **rewrite_dict}
                    for record in record_chunks
                    for rewrite_dict in (
                        record["requested_rewrite"]
                        if isinstance(record["requested_rewrite"], list)
                        else [record["requested_rewrite"]]
                    )
                ],
                hparams,
                return_orig_weights=False,
                **args_conserve_memory,
                **etc_args,
            )
        exec_time = time() - start
        cnt+=1
        print("Execution took", exec_time)
        # Evaluate new model
    
    # hs = get_module_input_output_at_words(
    #         edited_model,
    #         tok,
    #         hparams.layers[-1],
    #         context_templates=[request["template"] for request in eval_ds],
    #         words=[request["subject"] for request in eval_ds],
    #         module_template=hparams.layer_module_tmp,
    #         fact_token_strategy=hparams.fact_token,
    #     )[1].T
    # torch.save(hs, "post_edit_hs_memit.pt")
    start = time()
    gen_test_vars = [snips, vec]
    for record in ds:
        out_file = Path(case_result_template.format(num_edits, record["case_id"]))
        if out_file.exists():
            print(f"Skipping {out_file}; already exists")
            continue
        metrics = {
            "case_id": record["case_id"],
            "grouped_case_ids": case_ids,
            "num_edits": num_edits,
            "requested_rewrite": record["requested_rewrite"],
            "time": exec_time,
            "post": ds_eval_method(
                edited_model,
                tok,
                record,
                *(
                    gen_test_vars
                    if record["case_id"] % generation_test_interval == 0
                    else [None, None]
                ),  # Only test generation every generation_test_interval cases
            ),
        }
        # Dump metrics in .json
        with open(out_file, "w") as f:
            json.dump(metrics, f, indent=1)

        print("Evaluation took", time() - start)
    
    # Save edited model if requested (useful for attribute dataset downstream evaluation)
    if save_model:
        import pickle
        model_save_path = run_dir / "edited_model.pkl"
        print(f"Saving edited model to {model_save_path}")
        # Save model state dict and config
        save_data = {
            "model_state_dict": model.state_dict(),
            "model_name": model_name if isinstance(model_name, str) else model.config._name_or_path,
            "ds_name": ds_name,
            "num_edits": dataset_size_limit,
            "run_dir": str(run_dir),
        }
        with open(model_save_path, "wb") as f:
            pickle.dump(save_data, f)
        print(f"Model saved successfully!")

def get_project(model, tok, layer, hparams):
    force_recompute = False
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples
        if not force_recompute
        else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
    ).cpu()
    shared_visual = is_shared_visual_model(model)
    cov_reg = getattr(hparams, "nullspace_cov_reg", 0.0)
    if shared_visual and cov_reg > 0:
        cov = cov + cov_reg * torch.eye(cov.shape[0], dtype=cov.dtype)
        print(f"Applied shared-VL covariance regularization: +{cov_reg:g} I")

    S, U = torch.linalg.eigh(cov)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    null_dim = int(small_singular_indices.numel())
    total_dim = int(S.numel())
    null_ratio = null_dim / total_dim
    print(
        f"Layer {layer}: null-space dim {null_dim}/{total_dim} "
        f"({null_ratio:.1%}) with threshold={threshold:g}"
    )
    if shared_visual and null_ratio > 0.5:
        print("Warning: shared VL backbone still has a very large null space")
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T
def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["AlphaEdit","MEMIT_rect", "MEMIT_seq","MEMIT_prune", "MEMIT", "ROME", "FT", "MEND", "NSE", "PMET"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in "
        "results/<model_slug>/<alg_name>/<run_id> (model_slug from --model_name, "
        "e.g. AIDC-AI_Ovis-U1-3B), unless --results_model_dir is set. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--results_model_dir",
        type=str,
        default=None,
        help="Top-level folder under results/ for this run (default: sanitized --model_name, "
        "with '/' replaced by '_').",
    )
    parser.add_argument(
        "--ds_name",
        choices=["unike"],
        default="unike",
        help="Dataset to perform evaluations on.",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--downstream_eval_steps",
        type=int,
        default=0,
        help="If we want to do sequential editing or not",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save the edited model as a .pkl file for downstream evaluation (e.g., image generation).",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False, save_model=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        dir_name=None,
        results_model_dir=args.results_model_dir,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        save_model=args.save_model,
    )
