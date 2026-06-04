"""
Concurrent image generation for UniKE across Ovis-U1-3B, BLIP3o, and OmniGen2.
"""

import argparse
import gc
import io
import json
import os
import pickle
import sys
from pathlib import Path
import multiprocessing as mp

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from util.model_utils import (
    get_model_type, load_model_and_tok,
    _ensure_blip3o_path, _ensure_omnigen2_path,
)


class _CpuUnpickler(pickle.Unpickler):
    """Unpickler that forces all torch tensors to CPU, avoiding GPU memory waste."""
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def _load_pickle_cpu(path):
    """Load a pickle file with all torch tensors forced to CPU."""
    with open(path, "rb") as f:
        return _CpuUnpickler(f).load()


# ========== Ovis T2I ==========

def pipe_t2i_ovis(model, prompt, height, width, steps, cfg, seed=42):
    """Ovis-U1-3B text-to-image pipeline."""
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()

    gen_kwargs = dict(
        max_new_tokens=1024, do_sample=False,
        top_p=None, top_k=None, temperature=None, repetition_penalty=None,
        eos_token_id=text_tokenizer.eos_token_id,
        pad_token_id=text_tokenizer.pad_token_id,
        use_cache=True, height=height, width=width,
        num_steps=steps, seed=seed, img_cfg=0, txt_cfg=cfg,
    )

    blank_img = Image.new("RGB", (width, height), (255, 255, 255)).convert('RGB')

    uncond_prompt = "<image>\nGenerate an image."
    prompt_data, input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        uncond_prompt, [blank_img], generation_preface=None, return_labels=False,
        propagate_exception=False, multimodal_type='single_image',
        fix_sample_overall_length_navit=False,
    )
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(model.device)
    attention_mask = attention_mask.unsqueeze(0).to(model.device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(device=visual_tokenizer.device, dtype=torch.bfloat16)
    if grid_thws is not None:
        grid_thws = grid_thws.to(device=visual_tokenizer.device)
    with torch.inference_mode():
        no_both_cond = model.generate_condition(
            input_ids, pixel_values=pixel_values, attention_mask=attention_mask,
            grid_thws=grid_thws, **gen_kwargs
        )

    cond_prompt = "<image>\nDescribe the image by detailing the color, shape, size, texture, quantity, text, and spatial relationships of the objects:" + prompt
    prompt_data, input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        cond_prompt, [blank_img], generation_preface=None, return_labels=False,
        propagate_exception=False, multimodal_type='single_image',
        fix_sample_overall_length_navit=False,
    )
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(model.device)
    attention_mask = attention_mask.unsqueeze(0).to(model.device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(device=visual_tokenizer.device, dtype=torch.bfloat16)
    if grid_thws is not None:
        grid_thws = grid_thws.to(device=visual_tokenizer.device)

    target_size = (int(width), int(height))
    blank_img2, vae_pixel_values, cond_img_ids = model.visual_generator.process_image_aspectratio(blank_img, target_size)
    cond_img_ids[..., 0] = 1.0
    vae_pixel_values = vae_pixel_values.unsqueeze(0).to(device=model.device)

    with torch.inference_mode():
        cond = model.generate_condition(
            input_ids, pixel_values=pixel_values, attention_mask=attention_mask,
            grid_thws=grid_thws, **gen_kwargs
        )
        cond["vae_pixel_values"] = vae_pixel_values
        images = model.generate_img(cond=cond, no_both_cond=no_both_cond, no_txt_cond=None, **gen_kwargs)

    return images


# ========== BLIP3o T2I ==========

def init_blip3o_t2i(model, device="cpu"):
    """Load BLIP3o diffusion components once.

    device="cpu":  Components start on CPU, moved to GPU on demand during inference
                   (saves GPU memory on small GPUs like 48GB).
    device="cuda": Components loaded directly on GPU (faster on large GPUs).
    """
    from diffusers import UNet2DConditionModel, EulerDiscreteScheduler, AutoencoderKL

    model_path = model.config._name_or_path
    diffusion_path = model_path + "/diffusion-decoder"
    if not os.path.isdir(diffusion_path):
        from huggingface_hub import snapshot_download
        local_path = snapshot_download(repo_id=model_path)
        diffusion_path = os.path.join(local_path, "diffusion-decoder")

    unet = UNet2DConditionModel.from_pretrained(
        diffusion_path, subfolder="unet", torch_dtype=torch.bfloat16,
        use_safetensors=True, variant="bf16"
    )
    vae = AutoencoderKL.from_pretrained(
        diffusion_path, subfolder="vae", torch_dtype=torch.bfloat16,
        use_safetensors=True, variant="bf16"
    )
    if device == "cuda":
        unet = unet.to("cuda")
        vae = vae.to("cuda")
    scheduler = EulerDiscreteScheduler.from_pretrained(diffusion_path, subfolder="scheduler")
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    return {"unet": unet, "vae": vae, "scheduler": scheduler, "vae_scale_factor": vae_scale_factor}


def pipe_t2i_blip3o(model, tok, prompt, seed=42, t2i_components=None, offload=True):
    """BLIP3o text-to-image via diffusion decoder.

    offload=True:  Model→CPU after embedding generation, UNet/VAE→GPU for diffusion,
                   then reversed. Fits within 48GB.
    offload=False: Everything stays on GPU. Faster, needs >=80GB.
    """
    import random
    _ensure_blip3o_path()
    from BLIP3o.conversation import conv_templates
    from diffusers import UNet2DConditionModel, EulerDiscreteScheduler, AutoencoderKL

    height, width = 1024, 1024
    device = next(model.parameters()).device

    conv = conv_templates["qwen"].copy()
    conv.append_message(conv.roles[0], f"Please generate image based on the following caption: {prompt}")
    conv.append_message(conv.roles[1], None)
    text_prompt = conv.get_prompt()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    with torch.inference_mode():
        prompt_embeds = model.generate_image(text=[text_prompt], tokenizer=tok).to(torch.bfloat16).to(device)
        neg_embeds = model.generate_image(text=[" "], tokenizer=tok).to(torch.bfloat16).to(device)

    if offload:
        model.to("cpu")
        torch.cuda.empty_cache()

    unet = t2i_components["unet"]
    vae = t2i_components["vae"]
    scheduler = t2i_components["scheduler"]
    vae_scale_factor = t2i_components["vae_scale_factor"]

    if next(unet.parameters()).device.type == "cpu":
        unet.to(device)
    if next(vae.parameters()).device.type == "cpu":
        vae.to(device)

    combined_embeds = torch.cat([prompt_embeds, neg_embeds], dim=0)
    del prompt_embeds, neg_embeds

    time_ids = torch.LongTensor([height, width, 0, 0, height, width]).to(device)
    unet_added = {
        "time_ids": torch.cat([time_ids, time_ids], dim=0),
        "text_embeds": torch.mean(combined_embeds, dim=1),
    }

    num_steps = 50
    scheduler.set_timesteps(num_steps, device=device)
    shape = (1, unet.config.in_channels, height // vae_scale_factor, width // vae_scale_factor)
    latents = torch.randn(shape, device=device, dtype=torch.bfloat16)
    latents = latents * scheduler.init_noise_sigma

    with torch.inference_mode():
        for t in scheduler.timesteps:
            latent_input = torch.cat([latents] * 2)
            latent_input = scheduler.scale_model_input(latent_input, t)
            noise_pred = unet(latent_input, t, encoder_hidden_states=combined_embeds,
                              added_cond_kwargs=unet_added).sample
            noise_cond, noise_uncond = noise_pred.chunk(2)
            noise_pred = noise_uncond + 3.0 * (noise_cond - noise_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    latents = (1 / vae.config.scaling_factor) * latents
    with torch.inference_mode():
        image = vae.decode(latents.to(vae.dtype)).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")

    if offload:
        unet.to("cpu")
        vae.to("cpu")
        torch.cuda.empty_cache()
        model.to(device)

    return [Image.fromarray(image[0])]


# ========== OmniGen2 T2I ==========

def init_omnigen2_pipeline(edited_state_dict=None, offload=True):
    """
    Load OmniGen2 pipeline once. Call during worker init.
    If edited_state_dict is provided, injects edited weights into pipeline.mllm.

    offload=True:  CPU offload mode – only one component on GPU at a time (fits 48GB).
    offload=False: Full GPU mode – entire pipeline on GPU (needs >=80GB per worker).
    """
    _ensure_omnigen2_path()
    from OmniGen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
    from OmniGen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel

    model_path = "OmniGen2/OmniGen2"
    weight_dtype = torch.bfloat16

    pipeline = OmniGen2Pipeline.from_pretrained(
        model_path, torch_dtype=weight_dtype, trust_remote_code=True
    )
    pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=weight_dtype
    )

    if edited_state_dict is not None:
        pipeline.mllm.load_state_dict(edited_state_dict)

    if offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to("cuda")
    return pipeline


def pipe_t2i_omnigen2(pipeline, prompt, seed=42, steps=50):
    """OmniGen2 text-to-image using a pre-loaded pipeline."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    results = pipeline(
        prompt=prompt, input_images=None,
        width=1024, height=1024,
        num_inference_steps=steps, max_sequence_length=1024,
        text_guidance_scale=4.0, image_guidance_scale=2.0,
        cfg_range=(0.0, 1.0),
        negative_prompt="(((deformed))), blurry, bad anatomy, disfigured",
        num_images_per_prompt=1, generator=generator, output_type="pil",
    )
    return results.images


# ========== Worker ==========

def load_edited_model_for_images(edited_model_path, device="cuda"):
    """Load edited model for image generation.
    Moves state dict tensors to CPU first to avoid GPU memory waste from pickle deserialization.
    """
    save_data = _load_pickle_cpu(edited_model_path)

    model_name = save_data["model_name"]
    num_edits = save_data.get("num_edits", "unknown")
    model, tok = load_model_and_tok(model_name, device=device, dtype=torch.bfloat16)

    state_dict = save_data["model_state_dict"]
    del save_data
    converted = 0
    for key in state_dict:
        if state_dict[key].dtype == torch.float32:
            state_dict[key] = state_dict[key].to(torch.bfloat16)
            converted += 1
    model.load_state_dict(state_dict)
    del state_dict

    return model, tok, model_name, num_edits, converted


def worker_process(gpu_id, worker_id, task_queue, result_queue,
                   model_name, edited_model_path,
                   height, width, steps, cfg, seed, offload=True):
    """Worker process for image generation."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    model = None
    tok = None
    t2i_components = None  # BLIP3o UNet/VAE/scheduler
    omnigen2_pipeline = None  # OmniGen2 full pipeline

    model_type = get_model_type(model_name)

    if model_type == "omnigen2":
        offload_str = "CPU offload" if offload else "full GPU"
        print(f"[Worker {worker_id}] Loading OmniGen2 pipeline ({offload_str}) on GPU {gpu_id}...")
        edited_state_dict = None
        if edited_model_path:
            save_data = _load_pickle_cpu(edited_model_path)
            edited_state_dict = save_data["model_state_dict"]
            for key in edited_state_dict:
                if edited_state_dict[key].dtype == torch.float32:
                    edited_state_dict[key] = edited_state_dict[key].to(torch.bfloat16)
            del save_data
        omnigen2_pipeline = init_omnigen2_pipeline(edited_state_dict=edited_state_dict, offload=offload)
        del edited_state_dict
        torch.cuda.empty_cache()
        print(f"[Worker {worker_id}] OmniGen2 pipeline ready ({offload_str})")
    else:
        if edited_model_path:
            print(f"[Worker {worker_id}] Loading EDITED model on GPU {gpu_id}...")
            model, tok, _, num_edits, conv = load_edited_model_for_images(edited_model_path, "cuda")
            print(f"[Worker {worker_id}] Loaded ({num_edits} edits, {conv} tensors converted)")
        else:
            print(f"[Worker {worker_id}] Loading fresh model on GPU {gpu_id}...")
            model, tok = load_model_and_tok(model_name, device="cuda", dtype=torch.bfloat16)
            print(f"[Worker {worker_id}] Fresh model loaded")

        if model_type == "blip3o":
            device = "cuda" if not offload else "cpu"
            print(f"[Worker {worker_id}] Loading BLIP3o diffusion components (UNet/VAE) on {device}...")
            t2i_components = init_blip3o_t2i(model, device=device)
            print(f"[Worker {worker_id}] BLIP3o diffusion components ready")

    while True:
        task = task_queue.get()
        if task is None:
            break

        case_id, prompt, output_path, stage_key = task

        try:
            if os.path.exists(output_path):
                result_queue.put((case_id, stage_key, output_path, True, "exists"))
                continue

            if model_type == "ovis":
                images = pipe_t2i_ovis(model, prompt, height, width, steps, cfg, seed)
            elif model_type == "blip3o":
                images = pipe_t2i_blip3o(model, tok, prompt, seed, t2i_components=t2i_components, offload=offload)
            elif model_type == "omnigen2":
                images = pipe_t2i_omnigen2(omnigen2_pipeline, prompt, seed, steps=steps)
            else:
                result_queue.put((case_id, stage_key, output_path, False, "unsupported_model"))
                continue

            if images and len(images) > 0:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                images[0].save(output_path)
                result_queue.put((case_id, stage_key, output_path, True, "generated"))
            else:
                result_queue.put((case_id, stage_key, output_path, False, "generation_failed"))

        except Exception as e:
            import traceback
            traceback.print_exc()
            result_queue.put((case_id, stage_key, output_path, False, str(e)))

    print(f"[Worker {worker_id}] Shutting down")


def load_dataset(data_path):
    with open(data_path, 'r') as f:
        return json.load(f)


def load_reasoning_texts(reasoning_dir):
    reasoning_dir = Path(reasoning_dir)
    reasoning_texts = {}
    for json_file in reasoning_dir.glob("*.json"):
        if json_file.name.startswith("generation_summary"):
            continue
        with open(json_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    cid = item.get("case_id")
                    if cid is not None:
                        reasoning_texts[cid] = item
            elif isinstance(data, dict):
                cid = data.get("case_id")
                if cid is not None:
                    reasoning_texts[cid] = data
    return reasoning_texts


def main():
    parser = argparse.ArgumentParser(description="Concurrent Image Generation")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["pre_edit", "post_edit", "post_edit_no_reasoning"])
    parser.add_argument("--data_path", type=str, default="data/UniKE.json")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--edited_model_path", type=str, default=None)
    parser.add_argument("--reasoning_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="AIDC-AI/Ovis-U1-3B")
    parser.add_argument("--gpus", type=str, default="6,7")
    parser.add_argument("--models_per_gpu", type=int, default=1)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_offload", action="store_true",
                        help="Disable CPU offloading (faster on large GPUs >=80GB)")
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--primary_only", action="store_true",
                        help="Only generate stage_1 for faster Overall VQA evaluation")

    args = parser.parse_args()

    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    total_workers = len(gpu_ids) * args.models_per_gpu

    if args.mode == "post_edit" and (args.edited_model_path is None or args.reasoning_dir is None):
        print("ERROR: --edited_model_path and --reasoning_dir required for post_edit mode")
        sys.exit(1)
    if args.mode == "post_edit_no_reasoning" and args.edited_model_path is None:
        print("ERROR: --edited_model_path required for post_edit_no_reasoning mode")
        sys.exit(1)

    # Resolve model name (load on CPU to avoid wasting GPU memory)
    resolved_model_name = args.model_path
    if args.edited_model_path:
        save_data = _load_pickle_cpu(args.edited_model_path)
        resolved_model_name = save_data["model_name"]
        del save_data
        gc.collect()

    print(f"Configuration:")
    print(f"  Mode: {args.mode}")
    print(f"  Model: {resolved_model_name} ({get_model_type(resolved_model_name)})")
    print(f"  GPUs: {gpu_ids}, Workers: {total_workers}")
    print(f"  CPU offload: {not args.no_offload}")

    dataset = load_dataset(args.data_path)
    if args.start_idx is not None or args.end_idx is not None:
        start = args.start_idx or 0
        end = args.end_idx or len(dataset)
        dataset = dataset[start:end]
    else:
        start = 0

    reasoning_texts = {}
    if args.mode == "post_edit":
        reasoning_texts = load_reasoning_texts(args.reasoning_dir)
        print(f"Loaded {len(reasoning_texts)} reasoning texts")

    os.makedirs(args.output_dir, exist_ok=True)

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    tasks = []
    for i, item in enumerate(dataset):
        # Honor an explicit case_id on the item if present (used by the smoke
        # test, where a stratified subset preserves the original index of the
        # full dataset so that reasoning_*.json lookups still align).  Falls
        # back to positional indexing for the legacy full-dataset case.
        case_id = item.get("case_id", start + i)
        source = item.get("source", "")

        if source == "attribute":
            stage_keys = ["stage_1"] if args.primary_only else ["stage_1", "stage_2", "stage_3", "stage_4"]
        elif source == "relation":
            stage_keys = ["stage_1"]
        else:
            continue

        for stage_key in stage_keys:
            if stage_key not in item:
                continue
            stage = item[stage_key]
            original_prompt = stage.get("image_prompt", "")
            if args.mode in ("pre_edit", "post_edit_no_reasoning"):
                prompt = original_prompt
            else:
                rd = reasoning_texts.get(case_id, {})
                stage_rd = rd.get("stages", {}).get(stage_key, {})
                reasoning_output = stage_rd.get("reasoning_output", "")
                prompt = reasoning_output if reasoning_output else original_prompt
            if prompt:
                output_path = os.path.join(args.output_dir, f"case_{case_id}_{stage_key}.png")
                tasks.append((case_id, prompt, output_path, stage_key))

    print(f"Total tasks: {len(tasks)}")

    for task in tasks:
        task_queue.put(task)
    for _ in range(total_workers):
        task_queue.put(None)

    edited_model_path = args.edited_model_path if args.mode in ("post_edit", "post_edit_no_reasoning") else None

    workers = []
    worker_id = 0
    for gpu_id in gpu_ids:
        for _ in range(args.models_per_gpu):
            offload = not args.no_offload
            p = mp.Process(
                target=worker_process,
                args=(gpu_id, worker_id, task_queue, result_queue,
                      resolved_model_name, edited_model_path,
                      args.height, args.width, args.steps, args.cfg, args.seed, offload)
            )
            p.start()
            workers.append(p)
            worker_id += 1

    results = []
    pbar = tqdm(total=len(tasks), desc="Generating images")

    completed = 0
    while completed < len(tasks):
        result = result_queue.get()
        results.append(result)
        case_id, stage_key, path, success, msg = result
        pbar.set_postfix({"last": f"case_{case_id}_{stage_key}", "status": msg})
        pbar.update(1)
        completed += 1

    pbar.close()
    for p in workers:
        p.join()

    success_count = sum(1 for r in results if r[3])
    print(f"\nCompleted: {success_count}/{len(tasks)} images generated successfully")

    summary_path = os.path.join(args.output_dir, "generation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "mode": args.mode,
            "model": resolved_model_name,
            "model_type": get_model_type(resolved_model_name),
            "total_tasks": len(tasks),
            "success_count": success_count,
        }, f, indent=2)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
