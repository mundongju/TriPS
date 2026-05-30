#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TriPS — unified inference (root entry point)
============================================

Run posterior sampling with a *fixed, already-found* triadic schedule, for either:

  * --method TriPS-T : the training-free TEMPLATE schedule found by the TriPS-T
                       grid search (default values are the paper's confirmed
                       schedules, selectable per --dataset/--task; overridable).
  * --method TriPS-G : the GRPO-optimized schedule stored in a training checkpoint
                       (--grpo_ckpt), produced by TriPS_G/train_grpo_schedule_w_val.py.

Both methods share the same forward-operator setup, data loop, saving and metrics;
only the sampler + schedule source differ:
  TriPS-T -> sd3_sampler_TriPS_T.SD3FlowDPS.sample()       (computes the template curves)
  TriPS-G -> sd3_sampler_TriPS_G.SD3FlowDPS.sample_final(cfg/step/eta schedules)

Examples
--------
  # TriPS-T on the bundled demo images (super-resolution x8, DIV2K):
  python inference.py --method TriPS-T --dataset DIV2K --task sr_bicubic \
      --img_path demo_images/DIV2K --prompt_file demo_images/DIV2K/DIV2K_prompts_demo.txt \
      --workdir results/TriPS_T/DIV2K_srx8 --efficient_memory

  # TriPS-G with a trained checkpoint:
  python inference.py --method TriPS-G --task sr_bicubic --operator_imp SVD --deg_scale 8 \
      --img_path demo_images/DIV2K --prompt_file demo_images/DIV2K/DIV2K_prompts_demo.txt \
      --grpo_ckpt /path/to/grpo_schedule_ckpt_sr_bicubic_itXXXX.pt \
      --workdir results/TriPS_G/DIV2K_srx8 --efficient_memory
"""
import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid

import matplotlib
matplotlib.use("Agg")  # headless-safe (the TriPS-T sampler imports pyplot)

from util import set_seed, get_img_list, process_text
from eval import Metric

# ---------------------------------------------------------------------------
# TriPS-T final confirmed TEMPLATE schedules (match TriPS_G/init_load_file_fin/*.npz
# and TriPS_G/build_init_schedules.py). function_* in {linear, logarithm, exponential}.
# ---------------------------------------------------------------------------
TRIPS_T_PRESETS = {
    ("DIV2K", "sr_bicubic"):    dict(function_dc="linear",    function_cfg="logarithm",   function_sto="logarithm",
                                     step_scale=300.0, step_scale_2=100.0, stochasticity_weight=1.0, operator_imp="SVD", deg_scale=8),
    ("DIV2K", "deblur_gauss"):  dict(function_dc="linear",    function_cfg="logarithm",   function_sto="logarithm",
                                     step_scale=300.0, step_scale_2=200.0, stochasticity_weight=1.0, operator_imp="SVD", deg_scale=3),
    ("DIV2K", "deblur_motion"): dict(function_dc="linear",    function_cfg="exponential", function_sto="linear",
                                     step_scale=350.0, step_scale_2=150.0, stochasticity_weight=1.0, operator_imp="FFT", deg_scale=61),
    ("FFHQ",  "sr_bicubic"):    dict(function_dc="linear",    function_cfg="logarithm",   function_sto="logarithm",
                                     step_scale=250.0, step_scale_2=40.0,  stochasticity_weight=1.0, operator_imp="SVD", deg_scale=8),
    ("FFHQ",  "deblur_gauss"):  dict(function_dc="logarithm", function_cfg="logarithm",   function_sto="logarithm",
                                     step_scale=200.0, step_scale_2=100.0, stochasticity_weight=1.0, operator_imp="SVD", deg_scale=3),
    ("FFHQ",  "deblur_motion"): dict(function_dc="linear",    function_cfg="exponential", function_sto="linear",
                                     step_scale=350.0, step_scale_2=150.0, stochasticity_weight=1.0, operator_imp="FFT", deg_scale=61),
}

# inpainting masks (relative to repo root)
_MASK_DIR = Path(__file__).resolve().parent / "TriPS_G" / "exp" / "inp_masks"


# ===========================================================================
# GRPO ckpt -> (cfg, step, eta) schedules
# ===========================================================================
def load_grpo_schedules_from_ckpt(ckpt_path: str, NFE: int, device: torch.device, eps: float = 1e-6):
    from grpo_schedule import coeff_to_schedule, split_phi, ScheduleBounds
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "degree" not in ckpt or "bounds" not in ckpt:
        raise KeyError(f"ckpt missing 'degree'/'bounds': {ckpt_path}")
    degree = int(ckpt["degree"])
    bounds = ScheduleBounds(**ckpt["bounds"])
    d = degree + 1

    if "policy_state_dict" not in ckpt:
        raise KeyError(f"ckpt missing 'policy_state_dict': {ckpt_path}")
    sd = ckpt["policy_state_dict"]
    if ("log_alpha" in sd) and ("log_beta" in sd):
        alpha = sd["log_alpha"].float().exp().clamp_min(eps)
        beta = sd["log_beta"].float().exp().clamp_min(eps)
        phi_vec = (alpha / (alpha + beta)).clamp(eps, 1.0 - eps)
    elif "mu" in sd:
        phi_vec = sd["mu"].float()
    else:
        raise KeyError(f"policy_state_dict has no log_alpha/log_beta or mu: {ckpt_path}")

    if phi_vec.numel() != 3 * d:
        raise ValueError(f"phi dim mismatch: got {phi_vec.numel()}, expected {3 * d}")

    phi = phi_vec.to(device=device, dtype=torch.float32).unsqueeze(0)
    cfg_phi, step_phi, eta_phi = split_phi(phi, d, d, d)
    kw = dict(cfg_min=bounds.cfg_min, cfg_max=bounds.cfg_max,
              step_min=bounds.step_min, step_max=bounds.step_max, device=device)
    cfg_schedule = coeff_to_schedule(cfg_phi.squeeze(0), NFE, kind="cfg", **kw)
    step_schedule = coeff_to_schedule(step_phi.squeeze(0), NFE, kind="step", **kw)
    eta_schedule = coeff_to_schedule(eta_phi.squeeze(0), NFE, kind="eta", **kw)
    return cfg_schedule, step_schedule, eta_schedule, bounds, degree


# ===========================================================================
# Forward operator builder (shared)
# ===========================================================================
def build_operator(task, args, device, img_size):
    if task == "inpainting":
        from functions.svd_operators import Inpainting
        loaded = np.load(str(_MASK_DIR / "FFHQ_mask.npy"))
        mask = torch.from_numpy(loaded).to(device).reshape(-1)
        miss = torch.nonzero(mask == 0).long().reshape(-1) * 3
        missing = torch.cat([miss, miss + 1, miss + 2], dim=0)
        return Inpainting(3, img_size, missing, device)
    if task == "inpainting_DIV2K":
        from functions.svd_operators import Inpainting
        loaded = np.load(str(_MASK_DIR / "DIV2k_mask.npy"))
        mask = torch.from_numpy(loaded).to(device).reshape(-1)
        miss = torch.nonzero(mask == 0).long().reshape(-1) * 3
        missing = torch.cat([miss, miss + 1, miss + 2], dim=0)
        return Inpainting(3, img_size, missing, device)
    if task == "denoising":
        from functions.svd_operators import Denoising
        return Denoising(3, img_size, device)
    if task == "colorization":
        from functions.svd_operators import Colorization
        return Colorization(img_size, device)
    if task == "sr_averagepooling":
        from functions.svd_operators import SuperResolution
        return SuperResolution(3, img_size, int(args.deg_scale), device)
    if task == "sr_bicubic":
        factor = int(args.deg_scale)

        def bicubic_kernel(x, a=-0.5):
            if abs(x) <= 1:
                return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
            elif 1 < abs(x) < 2:
                return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
            return 0
        k = np.zeros((factor * 4))
        for i in range(factor * 4):
            x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
            k[i] = bicubic_kernel(x)
        k = k / np.sum(k)
        if args.operator_imp == "SVD":
            from functions.svd_operators import SRConv
            kernel = torch.from_numpy(k).float().to(device)
            return SRConv(kernel / kernel.sum(), 3, img_size, device, stride=factor)
        elif args.operator_imp == "FFT":
            from functions.fft_operators import Superres_fft, prepare_cubic_filter
            kk = prepare_cubic_filter(1 / factor)
            kernel = torch.from_numpy(kk).float().to(device)
            return Superres_fft(kernel / kernel.sum(), 3, img_size, device, stride=factor)
        raise NotImplementedError()
    if task == "deblur_gauss":
        sigma = args.deg_scale
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
        ker = [pdf(k) for k in range(-61 // 2, 61 // 2)]
        kernel = torch.Tensor(ker).to(device)
        if args.operator_imp == "SVD":
            from functions.svd_operators import Deblurring
            return Deblurring(kernel / kernel.sum(), 3, img_size, device)
        elif args.operator_imp == "FFT":
            from functions.fft_operators import Deblurring_fft
            return Deblurring_fft(kernel / kernel.sum(), 3, img_size, device)
        raise NotImplementedError()
    if task == "deblur_motion":
        # operator is rebuilt per-image (random kernel); validated below
        if args.operator_imp != "FFT":
            raise ValueError("deblur_motion requires --operator_imp FFT")
        return None
    raise ValueError(f"degradation type not supported: {task}")


@torch.no_grad()
def precompute_embeddings(args, prompts, solver):
    num = args.num_samples if args.num_samples > 0 else len(prompts)
    prompt_emb_set, pooled_emb_set = [], []
    for prompt in prompts[:num]:
        pe, pooled = solver.encode_prompt(prompt)
        prompt_emb_set.append(pe)
        pooled_emb_set.append(pooled)
    return prompt_emb_set, pooled_emb_set


def run(args):
    # -------- pick sampler + schedule source --------
    if args.method == "TriPS-T":
        from sd3_sampler_TriPS_T import get_solver
        solver = get_solver("TriPS_T")
    else:  # TriPS-G
        from sd3_sampler_TriPS_G import get_solver
        solver = get_solver("TriPS_G_test")
    solver.seed = args.seed

    # -------- prompts --------
    def sanitize(s, suffix):
        s = s.strip()
        return s[: -len(suffix)].strip() if suffix and s.endswith(suffix) else s

    prompts = process_text(prompt=args.prompt, prompt_file=args.prompt_file)
    if args.prompt_file is not None:
        prompts = [sanitize(p, args.prompt_suffix_to_remove) for p in prompts]

    solver.text_enc_1.to("cuda"); solver.text_enc_2.to("cuda"); solver.text_enc_3.to("cuda")
    if args.efficient_memory:
        with torch.no_grad():
            prompt_emb_set, pooled_emb_set = precompute_embeddings(args, prompts, solver)
            null_emb, null_pooled_emb = solver.encode_prompt([""])
        del solver.text_enc_1, solver.text_enc_2, solver.text_enc_3
        torch.cuda.empty_cache()
        prompt_embs = [[x, y] for x, y in zip(prompt_emb_set, pooled_emb_set)]
        null_embs = [null_emb, null_pooled_emb]
    else:
        prompt_embs = [[None, None]] * len(prompts)
        null_embs = [None, None]
    print("Prompts are processed.")

    solver.vae.to("cuda"); solver.transformer.to("cuda")
    device = solver.vae.device
    img_size = args.img_size

    A_funcs = build_operator(args.task, args, device, img_size)

    # -------- schedule source --------
    if args.method == "TriPS-G":
        cfg_schedule, step_schedule, eta_schedule, bounds, degree = load_grpo_schedules_from_ckpt(
            args.grpo_ckpt, NFE=args.NFE, device=device)
        print(f"[TriPS-G] ckpt={args.grpo_ckpt} | degree={degree} | "
              f"cfg[{cfg_schedule.min():.2f},{cfg_schedule.max():.2f}] "
              f"step[{step_schedule.min():.1f},{step_schedule.max():.1f}] "
              f"eta[{eta_schedule.min():.2f},{eta_schedule.max():.2f}]")
    else:
        print(f"[TriPS-T] template: dc={args.function_dc} cfg={args.function_cfg} sto={args.function_sto} | "
              f"step_scale={args.step_scale}->{args.step_scale_2} | sto_w={args.stochasticity_weight}")

    tf = transforms.Compose([transforms.Resize(img_size), transforms.CenterCrop(img_size), transforms.ToTensor()])

    for idx, path in enumerate(tqdm(get_img_list(args.img_path), desc=f"Solving [{args.method}]")):
        img_num = int(path.stem)
        img = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device) * 2 - 1

        if args.task == "deblur_motion":
            from functions.motionblur.motionblur import Kernel
            from functions.fft_operators import Deblurring_fft
            np.random.seed(seed=idx * 10)  # reproducible per-image kernel
            kernel = torch.from_numpy(Kernel(size=(args.deg_scale, args.deg_scale), intensity=0.5).kernelMatrix)
            A_funcs = Deblurring_fft(kernel / kernel.sum(), 3, img_size, solver.transformer.device)

        y = A_funcs.A(img)
        y = y + args.noise_std * torch.randn(y.shape, device=y.device,
                                             generator=torch.Generator(y.device).manual_seed(args.seed))

        prompt_i = prompts[idx] if len(prompts) > 1 else prompts[0]
        pe_i = prompt_embs[idx] if len(prompt_embs) > 1 else prompt_embs[0]

        proc_x0t, proc_x0y = [], []
        if args.method == "TriPS-T":
            out, proc_x0t = solver.sample(
                measurement=y, operator=A_funcs, task=args.task, prompts=prompt_i, NFE=args.NFE,
                img_shape=(img_size, img_size), cfg_scale=args.cfg_scale,
                prompt_embs=pe_i, null_embs=null_embs,
                step_scale=args.step_scale, step_scale_2=args.step_scale_2,
                inner_steps=args.inner_steps, sigma_y=args.noise_std,
                stochasticity_weight=args.stochasticity_weight, workdir=args.workdir,
                function_dc=args.function_dc, function_cfg=args.function_cfg, function_sto=args.function_sto,
                img_num=img_num)
        else:
            out, proc_x0t, proc_x0y = solver.sample_final(
                measurement=y, operator=A_funcs, task=args.task, prompts=prompt_i, NFE=args.NFE,
                img_shape=(img_size, img_size), cfg_scale=args.cfg_scale,
                prompt_embs=pe_i, null_embs=null_embs, inner_steps=args.inner_steps, sigma_y=args.noise_std,
                cfg_schedule=cfg_schedule, step_schedule=step_schedule, eta_schedule=eta_schedule)

        n = img_num
        if args.task in ["sr_bicubic", "inpainting", "inpainting_DIV2K", "deblur_gauss"]:
            save_image(A_funcs.At(y).reshape(img.shape), args.workdir / f"input1/{n:04d}.png", normalize=True)
        else:
            save_image(y.reshape(img.shape), args.workdir / f"input1/{n:04d}.png", normalize=True)
        save_image(out, args.workdir / f"recon/{n:04d}.png", normalize=True)
        save_image(img, args.workdir / f"label/{n:04d}.png", normalize=True)

        # process grids (guarded: TriPS-G's lists may be empty)
        for sub, proc in [("process_x0t", proc_x0t), ("process_x0y", proc_x0y)]:
            if proc:
                grid = make_grid(torch.cat(proc, dim=0), nrow=5, normalize=True)
                im = T.ToPILImage()(grid)
                im.resize((im.width // 2, im.height // 2)).save(args.workdir / sub / f"{n:05d}.png")

        if (idx + 1) == args.num_samples:
            break

    # -------- evaluation --------
    metric = Metric(["psnr", "lpips_flowdps", "lpips_flair", "ssim"])
    psnr_v, lpips_fd, lpips_fl, ssim_v = metric(args.workdir / "recon", args.workdir / "label")
    print("\n==== Evaluation ====")
    print(f"PSNR  : {psnr_v:.4f} dB")
    print(f"LPIPS (FlowDPS, 224): {lpips_fd:.4f}")
    print(f"LPIPS (FLAIR, full) : {lpips_fl:.4f}")
    print(f"SSIM  : {ssim_v:.4f}")
    with open(args.workdir / "eval_results.txt", "a") as f:
        f.write(f"method={args.method} task={args.task}\n")
        f.write(f"PSNR={psnr_v:.4f} LPIPS_FlowDPS={lpips_fd:.4f} LPIPS_FLAIR={lpips_fl:.4f} SSIM={ssim_v:.4f}\n")


def build_args():
    p = argparse.ArgumentParser(description="TriPS unified inference (TriPS-T / TriPS-G)")
    p.add_argument("--method", choices=["TriPS-T", "TriPS-G"], required=True)
    # data / problem
    p.add_argument("--dataset", choices=["DIV2K", "FFHQ"], default=None,
                   help="select TriPS-T preset schedule (TriPS-T only)")
    p.add_argument("--task", type=str, default="sr_bicubic")
    p.add_argument("--img_path", type=Path, required=True)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--prompt_file", type=str, default=None)
    p.add_argument("--prompt_suffix_to_remove", type=str, default=", high-resolution, 8k")
    p.add_argument("--num_samples", type=int, default=-1)
    p.add_argument("--deg_scale", type=int, default=None)
    p.add_argument("--operator_imp", type=str, default=None, help="SVD | FFT")
    # sampling
    p.add_argument("--img_size", type=int, default=768)
    p.add_argument("--NFE", type=int, default=28)
    p.add_argument("--cfg_scale", type=float, default=2.0)
    p.add_argument("--noise_std", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inner_steps", type=int, default=6)
    p.add_argument("--efficient_memory", action="store_true")
    p.add_argument("--workdir", type=Path, default=Path("results/TriPS_demo"))
    # TriPS-T template (default from preset)
    p.add_argument("--function_dc", type=str, default=None)
    p.add_argument("--function_cfg", type=str, default=None)
    p.add_argument("--function_sto", type=str, default=None)
    p.add_argument("--step_scale", type=float, default=None)
    p.add_argument("--step_scale_2", type=float, default=None)
    p.add_argument("--stochasticity_weight", type=float, default=None)
    # TriPS-G schedule
    p.add_argument("--grpo_ckpt", type=str, default=None, help="trained GRPO schedule .pt (TriPS-G)")
    args = p.parse_args()

    # fill TriPS-T defaults from preset
    if args.method == "TriPS-T":
        preset = TRIPS_T_PRESETS.get((args.dataset, args.task)) if args.dataset else None
        if preset is None and any(v is None for v in
                                  [args.function_dc, args.function_cfg, args.function_sto,
                                   args.step_scale, args.step_scale_2, args.stochasticity_weight]):
            raise SystemExit("TriPS-T: provide --dataset for a preset, or set all "
                             "--function_dc/--function_cfg/--function_sto/--step_scale/--step_scale_2/"
                             "--stochasticity_weight explicitly.")
        if preset is not None:
            for k in ["function_dc", "function_cfg", "function_sto",
                      "step_scale", "step_scale_2", "stochasticity_weight", "operator_imp", "deg_scale"]:
                if getattr(args, k) is None:
                    setattr(args, k, preset[k])
    else:  # TriPS-G
        if not args.grpo_ckpt:
            raise SystemExit("TriPS-G requires --grpo_ckpt")
        if args.operator_imp is None:
            args.operator_imp = "FFT" if args.task == "deblur_motion" else "SVD"
        if args.deg_scale is None:
            args.deg_scale = {"sr_bicubic": 8, "deblur_gauss": 3, "deblur_motion": 61}.get(args.task, 8)
    return args


if __name__ == "__main__":
    args = build_args()
    set_seed(args.seed)
    for sub in ["input1", "recon", "label", "process_x0t", "process_x0y"]:
        (args.workdir / sub).mkdir(parents=True, exist_ok=True)
    run(args)
