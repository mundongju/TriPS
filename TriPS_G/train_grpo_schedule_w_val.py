# train_grpo_schedule.py
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms

from sd3_sampler_TriPS_G_train import get_solver
from util import set_seed, get_img_list, process_text
from grpo_schedule import (
    poly_fit_ref_schedule, coeff_to_schedule,
    DiagBetaPolicy, split_phi, ScheduleBounds
)
from iqa_reward import (
    ModernIQAReward, ModernRewardConfig,
    CLIPIQAWrapper, QAlignWrapper
)
from typing import List
import numpy as np
import lpips
import random
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
import wandb
import re

def save_ckpt(
    ckpt_dir: Path,
    it: int,
    policy,
    optim,
    ref_alpha,
    ref_beta,
    bounds,
    args,
    history: dict | None = None,
    wandb_id: str | None = None,
    init_mu0: torch.Tensor | None = None,
    tag: str = "",
):
    """
    Full-state checkpoint for resuming training safely.
    Stores:
      - policy_state_dict / optim_state_dict
      - KL anchor (ref_alpha/ref_beta)
      - history (for plots)
      - init_mu0 (for overlay plots)
      - RNG states
      - wandb run id (so the same run can be resumed)
    """
    _ensure_dir(ckpt_dir)

    # args may contain Path objects -> stringify for safe serialization
    args_dict = {}
    for k, v in vars(args).items():
        args_dict[k] = str(v) if isinstance(v, Path) else v

    rng = {
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "np_rng_state": np.random.get_state(),
        "py_rng_state": random.getstate(),
    }

    ckpt = {
        "it": int(it),
        "policy_state_dict": policy.state_dict(),
        "optim_state_dict": optim.state_dict() if optim is not None else None,
        "ref_alpha": ref_alpha.detach().cpu(),
        "ref_beta": ref_beta.detach().cpu(),
        "degree": int(args.degree),
        "bounds": bounds.__dict__,
        "task": args.task,
        "NFE": int(args.NFE),
        "group_size": int(args.group_size),
        "lr": float(args.lr),
        "kl_beta": float(args.kl_beta),
        "clip_eps": float(args.clip_eps),
        "update_epochs": int(args.update_epochs),
        "args": args_dict,
        "history": history,
        "init_mu0": init_mu0.detach().cpu() if init_mu0 is not None else None,
        "wandb_id": wandb_id,
        "rng": rng,
    }

    ckpt_name = f"grpo_schedule_ckpt_{args.task}_it{it:04d}{tag}.pt"
    ckpt_path = ckpt_dir / ckpt_name
    torch.save(ckpt, ckpt_path)
    print(f"[Saved] ckpt -> {ckpt_path}")

def find_latest_ckpt(ckpt_dir: Path, task: str | None = None) -> Path | None:
    if not ckpt_dir.exists():
        return None
    pat = f"grpo_schedule_ckpt_{task}_it*.pt" if task else "grpo_schedule_ckpt_*_it*.pt"
    cand = list(ckpt_dir.glob(pat))
    if len(cand) == 0:
        return None

    def _parse_it(p: Path) -> int:
        m = re.search(r"_it(\d+)", p.stem)
        return int(m.group(1)) if m else -1

    cand.sort(key=_parse_it)
    return cand[-1]

def parse_it_from_path(p: Path) -> int:
    """Parse iteration number from checkpoint filename stem.

    Expected pattern: *_itXXXX*.pt
    """
    m = re.search(r"_it(\d+)", p.stem)
    return int(m.group(1)) if m else -1



def _move_optim_to_device(optim: torch.optim.Optimizer, device: torch.device):
    # After optim.load_state_dict, internal state tensors remain on their old device (often CPU).
    for state in optim.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_metrics_and_plots(workdir: Path, history: dict):
    """
    history keys: it, R_mean, PSNR, LPIPS, loss
    Saves:
      - train_metrics.npz
      - train_metrics_R_mean.png
      - train_metrics_PSNR.png
      - train_metrics_LPIPS.png
      - train_metrics_loss.png
      - train_metrics_all.png (4-in-1)
    """
    _ensure_dir(workdir)

    it = np.asarray(history["it"])
    np.savez(
        workdir / "train_metrics.npz",
        it=it,
        R_mean=np.asarray(history["R_mean"]),
        PSNR=np.asarray(history["PSNR"]),
        LPIPS=np.asarray(history["LPIPS"]),
        loss=np.asarray(history["loss"]),
    )

    def _plot_one(y, name, ylabel):
        plt.figure()
        plt.plot(it, y)
        plt.xlabel("iteration")
        plt.ylabel(ylabel)
        plt.title(name)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(workdir / f"train_metrics_{name}.png", dpi=200)
        plt.close()

    _plot_one(history["R_mean"], "R_mean", "R_mean")
    _plot_one(history["PSNR"], "PSNR", "PSNR")
    _plot_one(history["LPIPS"], "LPIPS", "LPIPS")
    _plot_one(history["loss"], "loss", "loss")

    # 4-in-1
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1); plt.plot(it, history["R_mean"]); plt.title("R_mean"); plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 2); plt.plot(it, history["PSNR"]);  plt.title("PSNR");  plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 3); plt.plot(it, history["LPIPS"]); plt.title("LPIPS"); plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 4); plt.plot(it, history["loss"]);  plt.title("loss");  plt.grid(True, alpha=0.3)
    for ax in plt.gcf().axes:
        ax.set_xlabel("iteration")
    plt.tight_layout()
    plt.savefig(workdir / "train_metrics_all.png", dpi=200)
    plt.close()

    # --------------------
    # Optional: validation metrics
    # --------------------
    if isinstance(history, dict) and ("val_it" in history) and len(history.get("val_it", [])) > 0:
        val_it = np.asarray(history["val_it"])

        # save raw arrays
        np.savez(
            workdir / "val_metrics.npz",
            it=val_it,
            R_mean=np.asarray(history.get("val_R_mean", [])),
            PSNR=np.asarray(history.get("val_PSNR", [])),
            SSIM=np.asarray(history.get("val_SSIM", [])),
            LPIPS=np.asarray(history.get("val_LPIPS", [])),
            CLIP_IQA=np.asarray(history.get("val_CLIP_IQA", [])),
            QALIGN=np.asarray(history.get("val_QALIGN", [])),
            distortion_score=np.asarray(history.get("val_distortion_score", [])),
            perception_score=np.asarray(history.get("val_perception_score", [])),
            num_images=np.asarray(history.get("val_num_images", [])),
        )

        def _plot_val(y, name, ylabel):
            if len(y) == 0:
                return
            plt.figure()
            plt.plot(val_it, y)
            plt.xlabel("iteration")
            plt.ylabel(ylabel)
            plt.title(f"val_{name}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(workdir / f"val_metrics_{name}.png", dpi=200)
            plt.close()

        _plot_val(history.get("val_R_mean", []), "R_mean", "val_R_mean")
        _plot_val(history.get("val_PSNR", []), "PSNR", "val_PSNR")
        _plot_val(history.get("val_SSIM", []), "SSIM", "val_SSIM")
        _plot_val(history.get("val_LPIPS", []), "LPIPS", "val_LPIPS")

def phi_to_schedules(phi_vec: torch.Tensor, degree: int, NFE: int, bounds, device: torch.device):
    d = degree + 1
    phi = phi_vec.detach().to(device).unsqueeze(0)  # [1, D_total]
    cfg_phi, step_phi, eta_phi = split_phi(phi, d, d, d)

    cfg_schedule = coeff_to_schedule(cfg_phi[0], NFE, "cfg",
                                     bounds.cfg_min, bounds.cfg_max, bounds.step_min, bounds.step_max, device=device)
    step_schedule = coeff_to_schedule(step_phi[0], NFE, "step",
                                      bounds.cfg_min, bounds.cfg_max, bounds.step_min, bounds.step_max, device=device)
    eta_schedule = coeff_to_schedule(eta_phi[0], NFE, "eta",
                                     bounds.cfg_min, bounds.cfg_max, bounds.step_min, bounds.step_max, device=device)
    return cfg_schedule, step_schedule, eta_schedule


def policy_to_schedules_mean(policy, degree: int, NFE: int, bounds, device: torch.device):
    """
    policy: DiagBetaPolicy (has log_alpha, log_beta)
    Returns representative schedules using Beta mean coefficients.
    """
    d = degree + 1

    alpha = policy.log_alpha.detach().exp().to(device)
    beta  = policy.log_beta.detach().exp().to(device)

    phi_mean = alpha / (alpha + beta)   # [D_total] in (0,1)
    phi = phi_mean.unsqueeze(0)         # [1, D_total]

    cfg_phi, step_phi, eta_phi = split_phi(phi, d, d, d)

    cfg_schedule = coeff_to_schedule(cfg_phi[0], NFE, "cfg",
                                     bounds.cfg_min, bounds.cfg_max, bounds.step_min, bounds.step_max, device=device)
    step_schedule = coeff_to_schedule(step_phi[0], NFE, "step",
                                      bounds.cfg_min, bounds.cfg_max, bounds.step_min, bounds.step_max, device=device)
    eta_schedule = coeff_to_schedule(eta_phi[0], NFE, "eta",
                                     bounds.cfg_min, bounds.cfg_max, bounds.step_min, bounds.step_max, device=device)
    return cfg_schedule, step_schedule, eta_schedule


def plot_schedule_overlay(workdir: Path, name: str, init_sched: torch.Tensor, final_sched: torch.Tensor):
    """
    Overlays init vs final schedule and saves png + npz
    """
    _ensure_dir(workdir)
    # x = np.arange(init_sched.numel())
    x = np.linspace(0.0, 1.0, 28)
    # x = np.load("./FFHQ_motion_deblur_sigma_eta_cfg_dc_ref_policy.npz", allow_pickle=True)["sigma"].astype(np.float64)

    init_np = init_sched.detach().float().cpu().numpy()
    final_np = final_sched.detach().float().cpu().numpy()

    np.savez(workdir / f"{name}_schedule_init_vs_final.npz", init=init_np, final=final_np)

    plt.figure()
    plt.plot(x, init_np, label="init")
    plt.plot(x, final_np, label="final")
    plt.xlabel("timestep (0..1)")
    plt.ylabel(name)
    plt.title(f"{name} schedule (init vs final)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(workdir / f"{name}_schedule_init_vs_final.png", dpi=200)
    plt.close()

def plot_schedule_overlay_eta(workdir: Path, name: str, init_sched: torch.Tensor, final_sched: torch.Tensor):
    """
    Overlays init vs final schedule and saves png + npz
    """
    _ensure_dir(workdir)
    # x = np.arange(init_sched.numel())
    x = np.linspace(0.0, 1.0, 28)
    # x = np.load("./FFHQ_motion_deblur_sigma_eta_cfg_dc_ref_policy.npz", allow_pickle=True)["sigma"].astype(np.float64)
    sigma = np.load("init_load_file_fin/FFHQ_motion_deblur_sigma_eta_cfg_dc_ref_policy_fin.npz", allow_pickle=True)["sigma"].astype(np.float64)
    # breakpoint()

    init_np = init_sched.detach().float().cpu().numpy()
    final_np = final_sched.detach().float().cpu().numpy()

    np.savez(workdir / f"{name}_schedule_init_vs_final.npz", init=init_np, final=final_np)

    plt.figure()
    plt.plot(x, init_np*sigma, label="init")
    plt.plot(x, final_np*sigma, label="final")
    # plt.plot(x, init_np, label="init")
    # plt.plot(x, final_np, label="final")
    plt.xlabel("timestep (0..1)")
    plt.ylabel(name)
    plt.title(f"{name} schedule (init vs final)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(workdir / f"{name}_schedule_init_vs_final.png", dpi=200)
    plt.close()

def _patch_starts(L: int, patch: int, stride: int) -> list[int]:
    """
    Returns start indices so that [start, start+patch) patches cover entire [0, L).
    Ensures last patch ends exactly at L.
    Example: L=768, patch=224, stride=224 -> [0,224,448,544]
    """
    assert patch <= L
    starts = []
    s = 0
    last = L - patch
    while s < last:
        starts.append(s)
        s += stride
    if len(starts) == 0 or starts[-1] != last:
        starts.append(last)
    return starts

@torch.no_grad()
def patch_lpips_vgg(
    lpips_fn,
    x01: torch.Tensor,
    y01: torch.Tensor,
    patch: int = 224,
    stride: int = 224,
) -> torch.Tensor:
    """
    Patch-based LPIPS over full image. Covers all regions by including the last patch aligned to the border.
    x01, y01: [1,3,H,W] in [0,1]
    returns scalar tensor
    """
    assert x01.ndim == 4 and y01.ndim == 4 and x01.shape == y01.shape
    B, C, H, W = x01.shape
    assert B == 1 and C == 3, "Expected [1,3,H,W]"
    assert H >= patch and W >= patch, "Image too small for patch size."

    # LPIPS expects inputs in [-1, 1]
    x = x01 * 2 - 1
    y = y01 * 2 - 1

    hs = _patch_starts(H, patch, stride)
    ws = _patch_starts(W, patch, stride)

    vals = []
    for i in hs:
        for j in ws:
            xp = x[:, :, i:i+patch, j:j+patch]
            yp = y[:, :, i:i+patch, j:j+patch]
            v = lpips_fn(xp, yp)   # shape [1,1,1,1] or [1,1]
            vals.append(v.reshape(-1)[0])

    return torch.stack(vals).mean()


def psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # x,y: [1,3,H,W] in [0,1]
    mse = torch.mean((x - y) ** 2)
    return 10.0 * torch.log10(1.0 / (mse + eps))

def _gaussian_window_2d(
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """2D Gaussian window for SSIM.

    Returns:
        window2d: [1,1,window_size,window_size]
    """
    assert window_size % 2 == 1, "window_size must be odd"
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size // 2)
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    window2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # [1,1,ws,ws]
    return window2d

@torch.no_grad()
def ssim(
    x01: torch.Tensor,
    y01: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    K1: float = 0.01,
    K2: float = 0.03,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Single-scale SSIM (mean over channel + spatial).

    Args:
        x01, y01: [1,3,H,W] in [0,1]
    """
    assert x01.ndim == 4 and y01.ndim == 4 and x01.shape == y01.shape
    B, C, H, W = x01.shape
    assert B == 1, "Expected batch=1"

    # Make sure window fits
    ws = int(window_size)
    ws = min(ws, H, W)
    if ws % 2 == 0:
        ws = max(1, ws - 1)
    ws = max(1, ws)

    # SSIM is typically computed in float32 for stability
    x = x01.float()
    y = y01.float()

    window = _gaussian_window_2d(ws, float(sigma), device=x.device, dtype=x.dtype)
    window = window.expand(C, 1, ws, ws).contiguous()  # [C,1,ws,ws]

    mu_x = F.conv2d(x, window, padding=ws // 2, groups=C)
    mu_y = F.conv2d(y, window, padding=ws // 2, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=ws // 2, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=ws // 2, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=ws // 2, groups=C) - mu_xy

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    num = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / (den + eps)
    return ssim_map.mean()

def _nanmean(x: torch.Tensor) -> torch.Tensor:
    """torch.nanmean fallback for older torch versions."""
    if hasattr(torch, 'nanmean'):
        return torch.nanmean(x)
    mask = ~torch.isnan(x)
    if mask.any():
        return x[mask].mean()
    return torch.tensor(float('nan'), device=x.device)

@torch.no_grad()
def precompute(args, prompts:List[str], solver) -> List[torch.Tensor]:
    prompt_emb_set = []
    pooled_emb_set = []
    num_samples = args.num_samples if args.num_samples > 0 else len(prompts)
    for prompt in prompts[:num_samples]:
        # prompt_emb, pooled_emb = solver.encode_prompt(prompt, batch_size=1)
        prompt_emb, pooled_emb = solver.encode_prompt(prompt)
        prompt_emb_set.append(prompt_emb)
        pooled_emb_set.append(pooled_emb)
    return prompt_emb_set, pooled_emb_set

def build_ref_schedules(NFE: int, conversion_step: int, device: torch.device, task: str = "sr", init_load_file=None):

    cfg_ref = np.load(init_load_file, allow_pickle=True)["cfg"].astype(np.float64)
    cfg_ref = torch.tensor(cfg_ref, device=device).float()

    step_ref = np.load(init_load_file, allow_pickle=True)["step_scale"].astype(np.float64)
    step_ref = torch.tensor(step_ref, device=device).float()

    eta_ref = np.load(init_load_file, allow_pickle=True)["eta"].astype(np.float64)
    eta_ref = torch.tensor(eta_ref, device=device).float()
    
    return cfg_ref, step_ref, eta_ref

def rollout_one(
    solver,
    A_funcs,
    y,
    gt_img01,
    prompt: str,
    NFE: int,
    img_size: int,
    cfg_schedule: torch.Tensor,
    step_schedule: torch.Tensor,
    eta_schedule: torch.Tensor,
    prompt_embs,
    null_embs,
    sigma_y: float,
    task,
    rewarder: ModernIQAReward,
    inner_steps,
):
    """Run one posterior sampling rollout and compute reward + metric_dict.

    Returns:
        reward: torch scalar tensor on solver device
        metric_dict: dict[str, float]
    """
    out01 = solver.sample_final(
        measurement=y,
        operator=A_funcs,
        task=task,
        prompts=prompt,
        NFE=NFE,
        img_shape=(img_size, img_size),
        prompt_embs=prompt_embs,
        null_embs=null_embs,
        sigma_y=sigma_y,
        # IMPORTANT: pass schedules
        cfg_schedule=cfg_schedule,
        step_schedule=step_schedule,
        eta_schedule=eta_schedule,
        inner_steps=inner_steps,
    )

    # Safety clamp (some solvers may slightly overshoot)
    out01 = out01.clamp(0, 1)

    with torch.no_grad():
        reward, metric_dict = rewarder.compute(out01, gt_img01, task, A_funcs)
    
    return reward, metric_dict

def run_validation(
    *,
    it: int,
    policy,
    solver,
    A_funcs,
    val_img_list: List[Path],
    prompts: List[str],
    prompt_embs,
    null_embs,
    tf,
    bounds,
    args,
    device: torch.device,
    rewarder: ModernIQAReward | None,
    lpips_fn,
    seed_sampling_fn,
    max_images: int = -1,
) -> dict:
    """Run validation on a dataset using the *current* policy mean schedule.

    Notes:
      - Uses reward_runs=1 by design (per user request) to keep validation fast.
      - Logs/returns dataset-mean metrics.
    """
    assert len(val_img_list) > 0, "val_img_list is empty"

    # Deterministic number of images
    n_total = len(val_img_list)
    n_eval = n_total if int(max_images) <= 0 else min(n_total, int(max_images))
    eval_paths = val_img_list[:n_eval]

    # Use representative (deterministic) schedules: Beta mean coefficients
    cfg_schedule, step_schedule, eta_schedule = policy_to_schedules_mean(
        policy, int(args.degree), int(args.NFE), bounds, device=device
    )

    # Accumulators (tensors on device)
    R_list = []
    psnr_list = []
    ssim_list = []
    msssim_list = []
    lpips_list = []
    clipiqa_list = []
    qalign_list = []
    nrviews_list = []
    dist_list = []
    perc_list = []

    pbar = tqdm(eval_paths, desc=f"Val@it{it}", leave=False)
    for j, path in enumerate(pbar):
        # Build operator per-image for motion deblur (kernel depends on seed)
        if args.task == 'deblur_motion':
            from functions.motionblur.motionblur import Kernel
            if args.operator_imp == 'FFT':
                from functions.fft_operators import Deblurring_fft
            else:
                raise ValueError("set operator_imp = FFT")

            np.random.seed(seed=j * 10)  # reproducibility per val image
            kernel = torch.from_numpy(
                Kernel(size=(args.deg_scale, args.deg_scale), intensity=0.5).kernelMatrix
            )
            A_funcs_i = Deblurring_fft(kernel / kernel.sum(), 3, args.img_size, solver.transformer.device)
        else:
            A_funcs_i = A_funcs

        gt = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(solver.vae.device)  # [0,1]
        gt = gt * 2 - 1
        gt01 = (gt / 2 + 0.5).clamp(0, 1)

        # Prompt selection (same logic as train)
        prompt_i = prompts[0] if len(prompts) == 1 else prompts[j % len(prompts)]
        prompt_emb_i = prompt_embs[j % len(prompt_embs)] if len(prompt_embs) > 1 else prompt_embs[0]

        # Measurement (deterministic noise, following train convention)
        y = A_funcs_i.A(gt)
        y = y + args.noise_std * torch.randn(
            y.shape,
            generator=torch.Generator(device).manual_seed(args.seed),
            device=device,
            dtype=y.dtype,
        )

        # Deterministic sampling seed per val-image (fixed across iterations for comparability)
        seed = int(args.seed + 777777 + j * 1009)
        seed_sampling_fn(seed)
        solver.seed = seed

        if args.reward_mode == "modern_iqa":
            if rewarder is None:
                raise RuntimeError("reward_mode=modern_iqa but rewarder is None")

            out01 = solver.sample_final(
                measurement=y,
                operator=A_funcs_i,
                task=args.task,
                prompts=prompt_i,
                NFE=args.NFE,
                img_shape=(args.img_size, args.img_size),
                prompt_embs=prompt_emb_i,
                null_embs=null_embs,
                sigma_y=args.noise_std,
                cfg_schedule=cfg_schedule,
                step_schedule=step_schedule,
                eta_schedule=eta_schedule,
                inner_steps=args.inner_steps,
            ).clamp(0, 1)

            r, md = rewarder.compute(out01, gt01)

            ps = torch.tensor(md.get("psnr", float('nan')), device=device)
            ms = torch.tensor(md.get("ms_ssim", float('nan')), device=device)
            lp = torch.tensor(md.get("lpips_patch", float('nan')), device=device)
            ci = torch.tensor(md.get("clip_iqa", float('nan')), device=device)
            qa = torch.tensor(md.get("qalign", float('nan')), device=device)
            nv = torch.tensor(md.get("nr_views_count", float('nan')), device=device)
            ds = torch.tensor(md.get("distortion_score", float('nan')), device=device)
            pc = torch.tensor(md.get("perception_score", float('nan')), device=device)
        else:
            # legacy baseline: PSNR - lpips_weight * patch LPIPS
            if lpips_fn is None:
                raise RuntimeError("reward_mode=psnr_lpips requires lpips_fn")

            out01 = solver.sample_final(
                measurement=y,
                operator=A_funcs_i,
                task=args.task,
                prompts=prompt_i,
                NFE=args.NFE,
                img_shape=(args.img_size, args.img_size),
                prompt_embs=prompt_emb_i,
                null_embs=null_embs,
                sigma_y=args.noise_std,
                cfg_schedule=cfg_schedule,
                step_schedule=step_schedule,
                eta_schedule=eta_schedule,
                inner_steps=args.inner_steps,
            ).clamp(0, 1)

            ps = psnr(out01.to(gt01.device), gt01)
            ms = torch.tensor(float('nan'), device=device)
            lp = patch_lpips_vgg(
                lpips_fn,
                out01.to(gt01.device),
                gt01,
                patch=int(getattr(args, "reward_lpips_patch", 224)),
                stride=int(getattr(args, "reward_lpips_stride", 224)),
            )
            r = ps - float(args.lpips_weight) * lp

            ci = torch.tensor(float('nan'), device=device)
            qa = torch.tensor(float('nan'), device=device)
            nv = torch.tensor(float('nan'), device=device)
            ds = torch.tensor(float('nan'), device=device)
            pc = torch.tensor(float('nan'), device=device)

        ss = ssim(out01.to(gt01.device), gt01)

        R_list.append(r)
        psnr_list.append(ps)
        ssim_list.append(ss)
        msssim_list.append(ms)
        lpips_list.append(lp)
        clipiqa_list.append(ci)
        qalign_list.append(qa)
        nrviews_list.append(nv)
        dist_list.append(ds)
        perc_list.append(pc)

        # quick progress info
        pbar.set_postfix({
            "R": float(r.detach().item()) if torch.is_tensor(r) else float(r),
            "PSNR": float(ps.detach().item()),
            "SSIM": float(ss.detach().item()),
        })

    def _stack_mean(xs: list[torch.Tensor]) -> float:
        if len(xs) == 0:
            return float('nan')
        return float(_nanmean(torch.stack(xs)).item())

    out = {
        "val_R_mean": _stack_mean(R_list),
        "val_PSNR": _stack_mean(psnr_list),
        "val_SSIM": _stack_mean(ssim_list),
        "val_MS_SSIM": _stack_mean(msssim_list),
        "val_LPIPS": _stack_mean(lpips_list),
        "val_CLIP_IQA": _stack_mean(clipiqa_list),
        "val_QALIGN": _stack_mean(qalign_list),
        "val_NRVIEWS": _stack_mean(nrviews_list),
        "val_distortion_score": _stack_mean(dist_list),
        "val_perception_score": _stack_mean(perc_list),
        "val_num_images": int(n_eval),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    # sampling params
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--NFE', type=int, default=28)
    parser.add_argument('--cfg_scale', type=float, default=2.0)
    parser.add_argument('--img_size', type=int, default=768)
    # workdir params
    parser.add_argument('--workdir', type=Path, default='workdir_TriPS_G_train_demo')
    parser.add_argument('--base_workdir', type=Path, default='workdir_251118_Inversion_mode1_inversion21')
    # data params
    parser.add_argument('--img_path', type=Path)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--inversion_prompt', type=str, default=None)
    parser.add_argument('--prompt_file', type=str, default=None)
    parser.add_argument('--prompt_file_val', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=-1)
    # problem params
    parser.add_argument('--task', type=str, default='sr_avgpool')
    parser.add_argument('--method', type=str, default='TriPS_G_train')
    parser.add_argument('--deg_scale', type=int, default=12)
    parser.add_argument('--noise_std', type=float, default=0.03)
    # solver params
    parser.add_argument('--inner_steps', type=int, default=6)
    parser.add_argument('--conversion_step', type=int, default=8)
    parser.add_argument('--efficient_memory',default=False, action='store_true')
    parser.add_argument('--attn_enforce', type=float, default=1.3)
    # Added for operator
    parser.add_argument(
        "--operator_imp", type=str, default="FFT", help="SVD | FFT"  # TODO: add CG support
    )

    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--mode', type=int, default=1)

    parser.add_argument('--prompt_suffix_to_remove', type=str, default=', high-resolution, 8k')

    # ========= GRPO params =========
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--kl_beta", type=float, default=0.01)

    # --- Adaptive KL control (to reference Beta distribution) ---
    # In this code, KL is computed as: KL( policy || ref ) where ref is fixed (ref_alpha/ref_beta).
    # The absolute scale depends on how DiagBetaPolicy.kl_to() is implemented (sum vs mean over dims),
    # so treat target_kl as a tunable hyperparameter.
    parser.add_argument("--target_kl", type=float, default=0.10,
                        help="Target KL(policy || ref) for adaptive KL penalty. Start around 0.08~0.12 for this setup.")
    parser.add_argument("--kl_beta_min", type=float, default=1e-5,
                        help="Minimum kl_beta when using adaptive KL.")
    parser.add_argument("--kl_beta_max", type=float, default=1.0,
                        help="Maximum kl_beta when using adaptive KL.")
    parser.add_argument("--kl_beta_up", type=float, default=2.0,
                        help="Multiplicative factor to increase kl_beta when KL is above the target band.")
    parser.add_argument("--kl_beta_down", type=float, default=1.5,
                        help="Multiplicative factor to decrease kl_beta when KL is below the target band (set 1.0 to disable).")
    parser.add_argument("--kl_tolerance", type=float, default=0.2,
                        help="Relative tolerance band around target_kl (0.2 -> [0.8*target, 1.2*target]).")
    parser.add_argument("--kl_warmup_iters", type=int, default=5,
                        help="Number of initial iterations before enabling kl_beta adaptation.")
    parser.add_argument("--kl_beta_adapt", dest="kl_beta_adapt", action="store_true",
                        help="Enable adaptive KL beta (default: enabled).")
    parser.add_argument("--no_kl_beta_adapt", dest="kl_beta_adapt", action="store_false",
                        help="Disable adaptive KL beta.")
    parser.set_defaults(kl_beta_adapt=True)
    parser.add_argument("--update_epochs", type=int, default=4)  # PPO-style multiple updates per rollout batch
    parser.add_argument('--init_load_file', type=Path, default='./ffhq_sr_sigma_eta_cfg_dc.npz')
    parser.add_argument("--lpips_weight", type=float, default=10)
    parser.add_argument("--kappa", type=float, default=10)


    # ========= Reward params (modern IQA, no pyiqa) =========
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="modern_iqa",
        choices=["modern_iqa", "psnr_lpips"],
        help="Reward mode. 'modern_iqa' uses [PSNR + patch-LPIPS + CLIP-IQA + Q-Align] with fixed scaling; 'psnr_lpips' uses PSNR - lpips_weight*patchLPIPS.",
    )

    # Distortion / Perception mixing (will be normalized to sum=1)
    parser.add_argument("--reward_distortion_weight", type=float, default=0.5)
    parser.add_argument("--reward_perception_weight", type=float, default=0.5)

    # ---- Distortion metrics ----
    parser.add_argument("--reward_use_psnr", dest="reward_use_psnr", action="store_true")
    parser.add_argument("--no_reward_use_psnr", dest="reward_use_psnr", action="store_false")
    parser.set_defaults(reward_use_psnr=True)
    parser.add_argument("--reward_psnr_min", type=float, default=20.0)
    parser.add_argument("--reward_psnr_max", type=float, default=40.0)
    parser.add_argument("--reward_w_psnr", type=float, default=1.0)

    # Optional MS-SSIM (distortion metric). Implemented inside iqa_reward.py (torch-only).
    parser.add_argument("--reward_use_msssim", dest="reward_use_msssim", action="store_true")
    parser.add_argument("--no_reward_use_msssim", dest="reward_use_msssim", action="store_false")
    parser.set_defaults(reward_use_msssim=False)
    parser.add_argument("--reward_msssim_min", type=float, default=0.80)
    parser.add_argument("--reward_msssim_max", type=float, default=1.00)
    parser.add_argument("--reward_w_msssim", type=float, default=1.0)

    # ---- Perception metrics ----
    parser.add_argument("--reward_use_lpips", dest="reward_use_lpips", action="store_true")
    parser.add_argument("--no_reward_use_lpips", dest="reward_use_lpips", action="store_false")
    parser.set_defaults(reward_use_lpips=False)
    parser.add_argument("--reward_lpips_patch", type=int, default=224)
    parser.add_argument("--reward_lpips_stride", type=int, default=224)
    parser.add_argument("--reward_w_lpips", type=float, default=1.0)

    parser.add_argument("--reward_use_clip_iqa", dest="reward_use_clip_iqa", action="store_true")
    parser.add_argument("--no_reward_use_clip_iqa", dest="reward_use_clip_iqa", action="store_false")
    parser.set_defaults(reward_use_clip_iqa=False)
    parser.add_argument("--reward_clip_model", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--reward_clip_prompts", type=str, default="quality,sharpness,noisiness")
    parser.add_argument("--reward_w_clip_iqa", type=float, default=1.0)

    parser.add_argument("--reward_use_qalign", dest="reward_use_qalign", action="store_true")
    parser.add_argument("--no_reward_use_qalign", dest="reward_use_qalign", action="store_false")
    parser.set_defaults(reward_use_qalign=False)
    parser.add_argument("--reward_qalign_model", type=str, default="q-future/one-align")
    parser.add_argument("--reward_qalign_task", type=str, default="quality", choices=["quality", "aesthetics"])
    parser.add_argument("--reward_w_qalign", type=float, default=1.0)

    # Shared resize for NR IQA (CLIP-IQA/Q-Align). 0 disables.
    parser.add_argument("--reward_iqa_resize", type=int, default=224)

    # NR-IQA view strategy for high-res images (e.g., 768x768)
    # 'resize': resize full image to reward_iqa_resize
    # 'center': native center crop (reward_nr_crop_native) -> resize to reward_iqa_resize
    # 'five'  : five-crop (center + 4 corners) at native res -> resize -> average
    parser.add_argument(
        "--reward_nr_view_mode",
        type=str,
        default="resize",
        choices=["resize", "center", "five"],
    )
    parser.add_argument(
        "--reward_nr_crop_native",
        type=int,
        default=0,
        help="Native crop size for NR-IQA views (e.g., 384/512). 0 disables cropping.",
    )

    # Q-Align model option
    parser.add_argument(
        "--qalign_attn_impl",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Transformers 'attn_implementation' for Q-Align/OneAlign.",
    )
    # =========================================================

    # ========= micro-batch / multi-run (variance reduction) =========
    parser.add_argument("--img_batch", "--img_batch_size", dest="img_batch", type=int, default=1,
                        help="number of different images per GRPO iteration (micro-batch). Sampler is still called sequentially with batch=1.")
    parser.add_argument("--reward_runs", "--multi_run", dest="reward_runs", type=int, default=1,
                        help="number of independent posterior sampling runs per rollout; rewards are averaged to reduce seed-induced variance.")
    # ==============================================================
    # ===============================

    # ========= validation params =========
    parser.add_argument(
        "--val_img_path",
        type=Path,
        default=None,
        help="Validation dataset image path/dir. If not set, validation is disabled.",
    )
    parser.add_argument(
        "--val_every",
        type=int,
        default=10,
        help="Run validation every N iterations (0 disables).",
    )
    parser.add_argument(
        "--val_max_images",
        type=int,
        default=-1,
        help="Max number of validation images per validation run (-1 uses all).",
    )
    # ===============================

    # ========= wandb params =========
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="grpo-schedule")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)     # run name
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name (alias). If set, overrides wandb_name when wandb_name is None.")
     # ===============================

    # ========= ckpt / run name params =========
    parser.add_argument("--ckpt_every", type=int, default=10, help="save ckpt every N iterations (0 disables)")
    parser.add_argument("--ckpt_dirname", type=str, default="ckpts", help="subdir under workdir to store ckpts")
    parser.add_argument("--resume", action="store_true", help="resume from latest ckpt under workdir/ckpt_dirname")
    parser.add_argument("--resume_ckpt", type=Path, default=None, help="explicit checkpoint path to resume from")
    parser.add_argument("--resume_strict", action="store_true", help="strict=True when loading policy_state_dict")
    # ==========================================

    args = parser.parse_args()

    # make sure workdir exists
    args.workdir = Path(args.workdir)
    args.base_workdir = Path(args.base_workdir)
    args.workdir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = args.workdir / args.ckpt_dirname
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- resume (pre-load ckpt meta for wandb/id & start_it) ----
    resume_state = None
    resume_path = None
    if args.resume_ckpt is not None:
        resume_path = Path(args.resume_ckpt)
    elif args.resume:
        # 1) try task-matching ckpt first
        resume_path = find_latest_ckpt(ckpt_dir, task=args.task)
        # 2) fallback: any task (common source of confusion when args.task differs)
        if resume_path is None:
            resume_path = find_latest_ckpt(ckpt_dir, task=None)
            if resume_path is not None:
                print(f"[Resume warn] No ckpt matched task='{args.task}'. Falling back to latest ckpt across all tasks: {resume_path.name}")

    if resume_path is not None and resume_path.exists():
        resume_state = torch.load(resume_path, map_location='cpu')
        it_in_ckpt = int(resume_state.get('it', -1))
        if it_in_ckpt < 0:
            it_in_ckpt = parse_it_from_path(resume_path)
        start_it = int(it_in_ckpt) + 1
        print(f"[Resume] ckpt={resume_path} (ckpt_it={it_in_ckpt}, start_it={start_it})")
    else:
        start_it = 0
        if args.resume or args.resume_ckpt is not None:
            print(f"[Resume] No checkpoint found under: {ckpt_dir} -> starting from it=0")


    # wandb
    wandb_run = None
    if args.use_wandb and args.wandb_mode != "disabled":
        wandb_id = None
        if resume_state is not None:
            wandb_id = resume_state.get("wandb_id", None)

        wandb_kwargs = dict(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=(args.wandb_name or args.run_name or str(args.workdir)),
            config=vars(args),
            dir=str(args.workdir),
            mode=args.wandb_mode,
        )
        if wandb_id is not None:
            wandb_kwargs["id"] = wandb_id
            wandb_kwargs["resume"] = "allow"

        wandb_run = wandb.init(**wandb_kwargs)

        # x-axis를 it로 고정
        wandb.define_metric("it")
        wandb.define_metric("*", step_metric="it")
    
    # ---- NEW: iteration log (CSV) ----
    log_path = args.workdir / "train_iter_log.csv"
    file_exists = log_path.exists()
    log_mode = "a" if (start_it > 0 and file_exists) else "w"
    log_f = open(log_path, log_mode, newline="")
    log_fields = [
        "it",
        "R_mean", "PSNR", "MS_SSIM", "LPIPS", "CLIP_IQA", "QALIGN", "NRVIEWS",
        "distortion_score", "perception_score",
        "R_max",
        "KL", "kl_beta", "kl_beta_next", "target_kl",
        "loss",
        "ratio_min", "ratio_mean", "ratio_max",
    ]
    log_writer = csv.DictWriter(log_f, fieldnames=log_fields, extrasaction="ignore")
    if log_mode == 'w' or (file_exists and log_path.stat().st_size == 0):
        log_writer.writeheader()

    # metric history
    history = {"it": [], "R_mean": [], "PSNR": [], "LPIPS": [], "loss": []}
    if resume_state is not None and isinstance(resume_state.get("history", None), dict):
        history = resume_state["history"]

    # workdir creation and seed setup
    set_seed(args.seed)
    solver = get_solver(args.method)
    solver.seed = args.seed

    ###### suffix remove ############################
    def sanitize_prompt(s: str, suffix: str) -> str:
        s = s.strip()
        if suffix and s.endswith(suffix):
            s = s[: -len(suffix)].strip()
        return s

    ################################## train prompts ##################################
    # load text prompts
    prompts = process_text(prompt=args.prompt, prompt_file=args.prompt_file)

    # suffix remove (DIV2K : prompt file // FFHQ : prompt)
    if args.prompt_file is not None:
        prompts = [sanitize_prompt(p, args.prompt_suffix_to_remove) for p in prompts]

    solver.text_enc_1.to('cuda')
    solver.text_enc_2.to('cuda')
    solver.text_enc_3.to('cuda')

    if args.efficient_memory:
        # precompute text embedding and remove encoders from GPU
        # This will allow us 1) fast inference 2) with lower memory requirement (<24GB)
        with torch.no_grad():
            prompt_emb_set, pooled_emb_set = precompute(args, prompts, solver)
            # null_emb, null_pooled_emb = solver.encode_prompt([''], batch_size=1)
            null_emb, null_pooled_emb = solver.encode_prompt([''])

        prompt_embs = [[x, y] for x, y in zip(prompt_emb_set, pooled_emb_set)]
        null_embs = [null_emb, null_pooled_emb]
    else:
        prompt_embs = [[None, None]] * len(prompts)
        inversion_prompt_embs = [[None, None]] * len(prompts)
        null_embs = [None, None]

    ################################## val prompts ##################################
    # load text prompts
    val_prompts = process_text(prompt=args.prompt, prompt_file=args.prompt_file_val)

    # suffix remove (DIV2K : prompt file // FFHQ : prompt)
    if args.prompt_file_val is not None:
        val_prompts = [sanitize_prompt(p, args.prompt_suffix_to_remove) for p in val_prompts]

    if args.efficient_memory:
        # precompute text embedding and remove encoders from GPU
        # This will allow us 1) fast inference 2) with lower memory requirement (<24GB)
        with torch.no_grad():
            prompt_emb_set_val, pooled_emb_set_val = precompute(args, val_prompts, solver)
            null_emb_val, null_pooled_emb_val = solver.encode_prompt([''])

        del solver.text_enc_1
        del solver.text_enc_2
        del solver.text_enc_3
        torch.cuda.empty_cache()
        prompt_embs_val = [[x, y] for x, y in zip(prompt_emb_set_val, pooled_emb_set_val)]
        null_embs_val = [null_emb_val, null_pooled_emb_val]
    else:
        prompt_embs = [[None, None]] * len(prompts)
        inversion_prompt_embs = [[None, None]] * len(prompts)
        null_embs = [None, None]

    print("Prompts are processed.")
    
    solver.vae.to('cuda')
    solver.transformer.to('cuda')

    #-----------------------problem setup------------------------
    device = solver.vae.device
    img_size = args.img_size

    if args.task == 'cs_walshhadamard':
        compress_by = round(1/args.deg_scale)
        from functions.svd_operators import WalshHadamardCS
        A_funcs = WalshHadamardCS(3, img_size, compress_by,
                                    torch.randperm(img_size ** 2, device=device), device)
    elif args.task == 'cs_blockbased':
        cs_ratio = args.deg_scale
        from functions.svd_operators import CS
        A_funcs = CS(3, img_size, cs_ratio, device)
    elif args.task == 'inpainting':
        from functions.svd_operators import Inpainting
        loaded = np.load("exp/inp_masks/FFHQ_mask.npy")
        mask = torch.from_numpy(loaded).to(device).reshape(-1)
        missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
        missing_g = missing_r + 1
        missing_b = missing_g + 1
        missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
        A_funcs = Inpainting(3, img_size, missing, device)
    elif args.task == 'inpainting_DIV2K':
        from functions.svd_operators import Inpainting
        loaded = np.load("exp/inp_masks/DIV2k_mask.npy")
        mask = torch.from_numpy(loaded).to(device).reshape(-1)
        missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
        missing_g = missing_r + 1
        missing_b = missing_g + 1
        missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
        A_funcs = Inpainting(3, img_size, missing, device)
    elif args.task == 'denoising':
        from functions.svd_operators import Denoising
        A_funcs = Denoising(3, img_size, device)
    elif args.task == 'colorization':
        from functions.svd_operators import Colorization
        A_funcs = Colorization(img_size, device)
    elif args.task == 'sr_averagepooling':
        blur_by = int(args.deg_scale)
        if args.operator_imp == 'SVD':
            from functions.svd_operators import SuperResolution
            A_funcs = SuperResolution(3, img_size, blur_by, device)
        else:
            raise NotImplementedError()

    elif args.task == 'sr_bicubic':
        factor = int(args.deg_scale)
        def bicubic_kernel(x, a=-0.5):
            if abs(x) <= 1:
                return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
            elif 1 < abs(x) and abs(x) < 2:
                return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
            else:
                return 0
        k = np.zeros((factor * 4))
        for i in range(factor * 4):
            x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
            k[i] = bicubic_kernel(x)
        k = k / np.sum(k)
        kernel = torch.from_numpy(k).float().to(device)
        
        if args.operator_imp == 'SVD':
            from functions.svd_operators import SRConv
            A_funcs = SRConv(kernel / kernel.sum(), 3, img_size, device, stride=factor)
        elif args.operator_imp == 'FFT':                
            from functions.fft_operators import Superres_fft, prepare_cubic_filter
            k = prepare_cubic_filter(1/factor)
            kernel = torch.from_numpy(k).float().to(device)
            A_funcs = Superres_fft(kernel / kernel.sum(), 3, img_size, device, stride=factor)
        else:
            raise NotImplementedError()

    elif args.task == 'deblur_uni':
        if args.operator_imp == 'SVD':
            from functions.svd_operators import Deblurring
            A_funcs = Deblurring(torch.Tensor([1 / 9] * 9).to(device), 3, img_size, device)
        elif args.operator_imp == 'FFT':
            from functions.fft_operators import Deblurring_fft
            A_funcs = Deblurring_fft(torch.Tensor([1 / 9] * 9).to(device), 3, img_size, device)
        else:
            raise NotImplementedError()

    elif args.task == 'deblur_gauss':
        # sigma = 50 # better make argument for kernel type
        sigma = args.deg_scale
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
        # kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(device) # clip it as in DDRM/DDNM code, but it makes more sense to use lower sigma with the line below
        # kernel = torch.Tensor([pdf(ii) for ii in range(-300,301,1)]).to(device)

        size = 61
        ker = []
        for k in range(-size//2, size//2):
            ker.append(pdf(k))
        kernel = torch.Tensor(ker).to(device)

        if args.operator_imp == 'SVD':
            from functions.svd_operators import Deblurring
            A_funcs = Deblurring(kernel / kernel.sum(), 3, img_size, device)
        elif args.operator_imp == 'FFT':
            from functions.fft_operators import Deblurring_fft
            A_funcs = Deblurring_fft(kernel / kernel.sum(), 3, img_size, device)
        else:
            raise NotImplementedError()

    elif args.task == 'deblur_aniso':
        sigma = 20
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
        kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(device)
        sigma = 1
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
        kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(device)

        if args.operator_imp == 'SVD':
            from functions.svd_operators import Deblurring2D
            A_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), 3, img_size, device)
        elif args.operator_imp == 'FFT':
            # unlike when using 'SVD' mode, here you can implement any 2D kernel that you want (not just seperable kernels)
            from functions.fft_operators import Deblurring_fft
            kernel = torch.matmul(kernel1[:,None],kernel2[None,:])
            A_funcs = Deblurring_fft(kernel / kernel.sum(), 3, img_size, device)
        else:
            raise NotImplementedError()

    elif args.task == 'deblur_motion':
        from functions.motionblur.motionblur import Kernel
        if args.operator_imp == 'FFT':
            from functions.fft_operators import Deblurring_fft
        else:
            raise ValueError("set operator_imp = FFT")

    else:
        raise ValueError("degradation type not supported")

    # ================= Reward / IQA modules =================
    # LPIPS is needed either for legacy psnr-lpips or when modern reward uses LPIPS.
    lpips_fn = None
    if args.reward_mode == "psnr_lpips" or (args.reward_mode == "modern_iqa" and args.reward_use_lpips):
        lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()

    rewarder = None
    if args.reward_mode == "modern_iqa":
        # Optional NR metrics (load only if enabled)
        clip_iqa = None
        if args.reward_use_clip_iqa:
            clip_prompts = tuple([p.strip() for p in args.reward_clip_prompts.split(',') if p.strip()])
            clip_iqa = CLIPIQAWrapper(
                device=device,
                model_name_or_path=args.reward_clip_model,
                prompts=clip_prompts,
            )

        qalign = None
        if args.reward_use_qalign:
            qalign = QAlignWrapper(
                model_name_or_path=args.reward_qalign_model,
                device=device,
                device_map="auto",
                attn_implementation=args.qalign_attn_impl,
            )

        cfg_reward = ModernRewardConfig(
            distortion_weight=float(args.reward_distortion_weight),
            perception_weight=float(args.reward_perception_weight),
            use_psnr=bool(args.reward_use_psnr),
            psnr_min=float(args.reward_psnr_min),
            psnr_max=float(args.reward_psnr_max),
            w_psnr=float(args.reward_w_psnr),
            use_msssim=bool(args.reward_use_msssim),
            msssim_min=float(args.reward_msssim_min),
            msssim_max=float(args.reward_msssim_max),
            w_msssim=float(args.reward_w_msssim),
            use_lpips=bool(args.reward_use_lpips),
            lpips_patch=int(args.reward_lpips_patch),
            lpips_stride=int(args.reward_lpips_stride),
            w_lpips=float(args.reward_w_lpips),
            use_clip_iqa=bool(args.reward_use_clip_iqa),
            w_clip_iqa=float(args.reward_w_clip_iqa),
            use_qalign=bool(args.reward_use_qalign),
            w_qalign=float(args.reward_w_qalign),
            qalign_task=str(args.reward_qalign_task),
            iqa_resize=(None if int(args.reward_iqa_resize) <= 0 else int(args.reward_iqa_resize)),
            nr_view_mode=str(args.reward_nr_view_mode),
            nr_crop_native=int(args.reward_nr_crop_native),
        )

        if cfg_reward.use_lpips and lpips_fn is None:
            raise RuntimeError("LPIPS is required but lpips_fn is None. Check reward_mode/reward_use_lpips.")

        rewarder = ModernIQAReward(
            device=device,
            lpips_fn=lpips_fn,
            cfg=cfg_reward,
            clip_iqa=clip_iqa,
            qalign=qalign,
        )

        print("[Reward] mode=modern_iqa | distortion_weight=%.3f perception_weight=%.3f" % (
            cfg_reward.distortion_weight, cfg_reward.perception_weight
        ))
        print("[Reward] use_psnr=%s use_lpips=%s use_clip_iqa=%s use_qalign=%s" % (
            cfg_reward.use_psnr, cfg_reward.use_lpips, cfg_reward.use_clip_iqa, cfg_reward.use_qalign
        ))
    else:
        print("[Reward] mode=psnr_lpips (legacy): reward = PSNR - lpips_weight * patchLPIPS")

    # =========================================================

    # Reference schedules + polynomial fit initialization
    bounds = ScheduleBounds(cfg_min=1.0, cfg_max=8.0, step_min=10.0, step_max=400.0)

    cfg_ref, step_ref, eta_ref = build_ref_schedules(args.NFE, args.conversion_step, device=device, init_load_file=args.init_load_file)

    cfg_coeff0 = poly_fit_ref_schedule(cfg_ref, args.degree, "cfg", bounds.cfg_min, bounds.cfg_max, bounds.step_min, bounds.step_max)
    step_coeff0 = poly_fit_ref_schedule(step_ref, args.degree, "step", bounds.cfg_min, bounds.cfg_max, bounds.step_min, bounds.step_max)
    eta_coeff0 = poly_fit_ref_schedule(eta_ref.clamp(1e-3, 1-1e-3), args.degree, "eta", bounds.cfg_min, bounds.cfg_max, bounds.step_min, bounds.step_max)

    init_mu = torch.cat([cfg_coeff0, step_coeff0, eta_coeff0], dim=0).to(device)
    init_mu = init_mu.clamp(1e-4, 1 - 1e-4)
    init_mu0 = init_mu.detach().clone()
    # init_log_std = torch.full_like(init_mu, -2.0)  # small exploration

    # kappa = 50.0  # exploration scale
    kappa = args.kappa
    init_alpha = (init_mu * kappa).clamp_min(1e-3)
    init_beta  = ((1 - init_mu) * kappa).clamp_min(1e-3)
    policy = DiagBetaPolicy(init_alpha, init_beta).to(device)

    # --- NEW: policy_old (behavior policy snapshot) ---
    policy_old = DiagBetaPolicy(init_alpha.detach().clone(), init_beta.detach().clone()).to(device)
    policy_old.load_state_dict(policy.state_dict())
    policy_old.eval()
    for p in policy_old.parameters():
        p.requires_grad_(False)
    # -----------------------------------------------

    # ref_mu = init_mu.detach().clone()
    # ref_log_std = init_log_std.detach().clone()
    ref_alpha = init_alpha.detach().clone()
    ref_beta  = init_beta.detach().clone()

    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)
    # ---- apply resume state (policy/optim/ref/history/rng) ----
    if resume_state is not None:
        # warn on mismatches (do not auto-override)
        if int(resume_state.get("degree", args.degree)) != int(args.degree):
            print(f"[Resume warn] degree mismatch: ckpt={resume_state.get('degree')} vs args={args.degree}")
        if int(resume_state.get("NFE", args.NFE)) != int(args.NFE):
            print(f"[Resume warn] NFE mismatch: ckpt={resume_state.get('NFE')} vs args={args.NFE}")
        if str(resume_state.get("task", args.task)) != str(args.task):
            print(f"[Resume warn] task mismatch: ckpt={resume_state.get('task')} vs args={args.task}")

        # Restore adaptive KL beta if it was saved in the checkpoint (for exact resumption).
        if resume_state.get("kl_beta", None) is not None:
            args.kl_beta = float(resume_state["kl_beta"])
            print(f"[Resume] restored kl_beta={args.kl_beta}")

        # Load policy + optimizer
        if resume_state.get("policy_state_dict", None) is not None:
            policy.load_state_dict(resume_state["policy_state_dict"], strict=bool(args.resume_strict))
        # optimizer
        if resume_state.get("optim_state_dict", None) is not None:
            optim.load_state_dict(resume_state["optim_state_dict"])
            _move_optim_to_device(optim, device)

        # KL anchors
        if resume_state.get("ref_alpha", None) is not None:
            ref_alpha = resume_state["ref_alpha"].to(device)
        if resume_state.get("ref_beta", None) is not None:
            ref_beta = resume_state["ref_beta"].to(device)

        # init_mu0 for overlay plots (optional)
        if resume_state.get("init_mu0", None) is not None:
            init_mu0 = resume_state["init_mu0"].to(device)

        # RNG restore (optional)
        rng = resume_state.get("rng", None)
        if isinstance(rng, dict):
            try:
                torch.set_rng_state(rng["torch_rng_state"])
                if torch.cuda.is_available() and rng.get("cuda_rng_state_all", None) is not None:
                    torch.cuda.set_rng_state_all(rng["cuda_rng_state_all"])
                np.random.set_state(rng["np_rng_state"])
                random.setstate(rng["py_rng_state"])
            except Exception as e:
                print(f"[Resume warn] RNG restore failed: {e}")

        # refresh behavior policy snapshot
        policy_old.load_state_dict(policy.state_dict())

    # Data loader-ish (batch=1)
    tf = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor()
    ])

    img_list = list(get_img_list(args.img_path))
    if len(img_list) == 0:
        raise ValueError(f"No images found under: {args.img_path}")

    # ================= Validation dataset =================
    val_img_list = None
    if getattr(args, "val_img_path", None) is not None:
        val_img_list = list(get_img_list(args.val_img_path))
        if len(val_img_list) == 0:
            print(f"[Val warn] No images found under: {args.val_img_path} -> validation disabled")
            val_img_list = None

    if val_img_list is not None and int(getattr(args, "val_every", 0)) <= 0:
        print("[Val warn] val_img_path is set but val_every<=0 -> validation disabled")
        val_img_list = None

    # ---- validation log (CSV) ----
    val_log_f = None
    val_log_writer = None
    val_log_path = args.workdir / "val_iter_log.csv"
    if val_img_list is not None:
        val_file_exists = val_log_path.exists()
        val_log_mode = "a" if (start_it > 0 and val_file_exists) else "w"
        val_log_f = open(val_log_path, val_log_mode, newline="")
        val_log_fields = [
            "it",
            "val_num_images",
            "val_R_mean",
            "val_PSNR",
            "val_SSIM",
            "val_MS_SSIM",
            "val_LPIPS",
            "val_CLIP_IQA",
            "val_QALIGN",
            "val_NRVIEWS",
            "val_distortion_score",
            "val_perception_score",
        ]
        val_log_writer = csv.DictWriter(val_log_f, fieldnames=val_log_fields, extrasaction="ignore")
        if val_log_mode == 'w' or (val_file_exists and val_log_path.stat().st_size == 0):
            val_log_writer.writeheader()

        # ensure history has validation keys (for resume compatibility)
        history.setdefault("val_it", [])
        history.setdefault("val_R_mean", [])
        history.setdefault("val_PSNR", [])
        history.setdefault("val_SSIM", [])
        history.setdefault("val_LPIPS", [])
        history.setdefault("val_CLIP_IQA", [])
        history.setdefault("val_QALIGN", [])
        history.setdefault("val_distortion_score", [])
        history.setdefault("val_perception_score", [])
        history.setdefault("val_num_images", [])
    # ======================================================

    # ---- NEW: micro-batch + multi-run configs ----
    img_batch = max(1, int(getattr(args, "img_batch", 1)))
    img_batch = min(img_batch, len(img_list))
    reward_runs = max(1, int(getattr(args, "reward_runs", 1)))

    # NEW: epoch-style shuffle state (reproducible)
    N = len(img_list)
    shuffle_gen = torch.Generator(device="cpu").manual_seed(int(getattr(args, "seed", 0)) + 12345)
    perm = torch.randperm(N, generator=shuffle_gen).tolist()
    ptr = 0  # points to next unread index in perm

    def _seed_sampling(seed: int):
        """Seed torch/numpy to control stochasticity inside posterior sampling."""
        # torch seeds are 64-bit; numpy needs 32-bit
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32 - 1))

    def _make_seed(it: int, img_idx: int, g: int, run: int) -> int:
        # Deterministic seed schedule (reproducible) for posterior sampling noise
        return int(args.seed + it * 1000003 + img_idx * 1009 + g * 17 + run)

    pbar = tqdm(range(start_it, args.iters), desc="GRPO schedule training", total=args.iters, initial=start_it)
    for it in pbar:
        
        # Helpful one-time print to avoid confusion about whether resume worked.
        if it == start_it:
            print(f"[Train] entering loop at it={it} (start_it={start_it})")
            
        # ================================
        # 0) Snapshot current policy -> policy_old
        # ================================
        policy_old.load_state_dict(policy.state_dict())

        # pick multiple images for this iteration (micro-batch; processed sequentially)
        # base = (it * img_batch) % len(img_list)
        # idxs = [(base + b) % len(img_list) for b in range(img_batch)]

        # NEW: take next img_batch indices from the current epoch permutation
        if ptr + img_batch <= N:
            idxs = perm[ptr:ptr + img_batch]
            ptr += img_batch
        else:
            # not enough left in this epoch -> consume the rest, reshuffle, then take the remainder
            idxs = perm[ptr:]
            need = img_batch - len(idxs)

            perm = torch.randperm(N, generator=shuffle_gen).tolist()  # new epoch shuffle
            ptr = 0
            idxs += perm[ptr:ptr + need]
            ptr += need

        # ================================
        # 1) Sample group actions using policy_old (behavior policy)
        #    (IMPORTANT: sample all actions first, then change seeds for rollouts)
        # ================================
        _seed_sampling(args.seed + it * 97)
        with torch.no_grad():
            # Sample *one* group of actions (phi) and share it across all images in this micro-batch.
            # This matches the intent of reducing "batch-induced" variance: the policy is evaluated on multiple images
            # under the same set of candidate schedules.
            phi, _ = policy_old.sample(img_batch * args.group_size)   # [B*G, D]
            logp_old = policy_old.log_prob(phi)                       # [B*G]

        # build schedules per group element (shared across images)
        d = args.degree + 1
        cfg_phi_flat, step_phi_flat, eta_phi_flat = split_phi(phi, d, d, d)  # each [G, d]

        cfg_schedules_flat = []
        step_schedules_flat = []
        eta_schedules_flat = []
        for k in range(img_batch * args.group_size):
            cfg_schedules_flat.append(
                coeff_to_schedule(cfg_phi_flat[k], args.NFE, "cfg",
                                  bounds.cfg_min, bounds.cfg_max, bounds.step_min, bounds.step_max, device=device)
            )
            step_schedules_flat.append(
                coeff_to_schedule(step_phi_flat[k], args.NFE, "step",
                                  bounds.cfg_min, bounds.cfg_max, bounds.step_min, bounds.step_max, device=device)
            )
            eta_schedules_flat.append(
                coeff_to_schedule(eta_phi_flat[k], args.NFE, "eta",
                                  bounds.cfg_min, bounds.cfg_max, bounds.step_min, bounds.step_max, device=device)
            )
        cfg_schedules = torch.stack(cfg_schedules_flat, dim=0).view(img_batch, args.group_size, -1)    # [B,G,NFE]
        step_schedules = torch.stack(step_schedules_flat, dim=0).view(img_batch, args.group_size, -1)  # [B,G,NFE]
        eta_schedules = torch.stack(eta_schedules_flat, dim=0).view(img_batch, args.group_size, -1)    # [B,G,NFE]

        if it == 0:
            print('-'*10)
            print((cfg_schedules[0,0]-cfg_ref).norm().item())
            print((step_schedules[0,0]-step_ref).norm().item())
            print((eta_schedules[0,0]-eta_ref).norm().item())
            print('-'*10)

            # cfg_init, step_init, eta_init = phi_to_schedules(init_mu0, args.degree, args.NFE, bounds, device=device)
            # plot_schedule_overlay(args.workdir, "cfg", cfg_init, cfg_init)
            # plot_schedule_overlay(args.workdir, "step", step_init, step_init)
            # plot_schedule_overlay_eta(args.workdir, "eta", eta_init, eta_init)
            # print(f"[Saved plots] metrics + schedules -> {args.workdir}")
            # breakpoint()

        rewards = torch.empty((img_batch, args.group_size), device=device)
        psnrs = torch.full((img_batch, args.group_size), float('nan'), device=device)
        msssims = torch.full((img_batch, args.group_size), float('nan'), device=device)
        lpipss = torch.full((img_batch, args.group_size), float('nan'), device=device)
        clipiqas = torch.full((img_batch, args.group_size), float('nan'), device=device)
        qaligns = torch.full((img_batch, args.group_size), float('nan'), device=device)
        nrviews = torch.full((img_batch, args.group_size), float('nan'), device=device)
        dist_scores = torch.full((img_batch, args.group_size), float('nan'), device=device)
        perc_scores = torch.full((img_batch, args.group_size), float('nan'), device=device)

        # ================================
        # 1.5) Rollout (image-by-image, because sampler assumes batch=1)
        # ================================
        for b, idx in enumerate(idxs):
            path = img_list[idx]

            # Build operator per-image for motion deblur (kernel depends on seed)
            if args.task == 'deblur_motion':
                from functions.motionblur.motionblur import Kernel
                if args.operator_imp == 'FFT':
                    from functions.fft_operators import Deblurring_fft
                else:
                    raise ValueError("set operator_imp = FFT")

                np.random.seed(seed=idx * 10)  # reproducibility per image
                kernel = torch.from_numpy(
                    Kernel(size=(args.deg_scale, args.deg_scale), intensity=0.5).kernelMatrix
                )
                A_funcs_i = Deblurring_fft(kernel / kernel.sum(), 3, args.img_size, solver.transformer.device)
            else:
                A_funcs_i = A_funcs

            gt = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(solver.vae.device)  # [0,1]
            gt = gt * 2 - 1  # match your solve_ours convention
            gt01 = (gt / 2 + 0.5).clamp(0, 1)

            # prompt selection
            prompt_i = prompts[0] if len(prompts) == 1 else prompts[idx % len(prompts)]
            prompt_emb_i = prompt_embs[idx % len(prompt_embs)] if len(prompt_embs) > 1 else prompt_embs[0]

            # measurement (keep measurement noise deterministic per image)
            y = A_funcs_i.A(gt)
            y = y + args.noise_std * torch.randn(y.shape, generator=torch.Generator(device).manual_seed(args.seed), device=device, dtype=y.dtype)

            for g in range(args.group_size):
                cfg_schedule = cfg_schedules[b, g]
                step_schedule = step_schedules[b, g]
                eta_schedule = eta_schedules[b, g]

                # ---- multi-run reward averaging (different posterior sampling noise per run) ----
                r_runs = []
                ps_runs = []
                ms_runs = []
                lp_runs = []
                clip_runs = []
                qalign_runs = []
                nrview_runs = []
                dist_runs = []
                perc_runs = []
                for rr in range(reward_runs):
                    seed = _make_seed(it, idx, g, rr)
                    _seed_sampling(seed)
                    solver.seed = seed

                    if args.reward_mode == "modern_iqa":
                        if rewarder is None:
                            raise RuntimeError("reward_mode=modern_iqa but rewarder is None")

                        r, md = rollout_one(
                            solver=solver,
                            A_funcs=A_funcs_i,
                            task=args.task,
                            y=y,
                            gt_img01=gt01,
                            prompt=prompt_i,
                            prompt_embs=prompt_emb_i,
                            null_embs=null_embs,
                            NFE=args.NFE,
                            img_size=args.img_size,
                            cfg_schedule=cfg_schedule,
                            step_schedule=step_schedule,
                            eta_schedule=eta_schedule,
                            sigma_y=args.noise_std,
                            rewarder=rewarder,
                            inner_steps=args.inner_steps,
                        )

                        ps = torch.tensor(md.get("psnr", float('nan')), device=device)
                        ms = torch.tensor(md.get("ms_ssim", float('nan')), device=device)
                        lp = torch.tensor(md.get("lpips_patch", float('nan')), device=device)
                        ci = torch.tensor(md.get("clip_iqa", float('nan')), device=device)
                        qa = torch.tensor(md.get("qalign", float('nan')), device=device)
                        nv = torch.tensor(md.get("nr_views_count", float('nan')), device=device)
                        ds = torch.tensor(md.get("distortion_score", float('nan')), device=device)
                        pc = torch.tensor(md.get("perception_score", float('nan')), device=device)

                    else:
                        # legacy baseline: PSNR - lpips_weight * patch LPIPS
                        if lpips_fn is None:
                            raise RuntimeError("reward_mode=psnr_lpips requires lpips_fn")

                        out01 = solver.sample_final(
                            measurement=y,
                            operator=A_funcs_i,
                            task=args.task,
                            prompts=prompt_i,
                            NFE=args.NFE,
                            img_shape=(args.img_size, args.img_size),
                            prompt_embs=prompt_emb_i,
                            null_embs=null_embs,
                            sigma_y=args.noise_std,
                            cfg_schedule=cfg_schedule,
                            step_schedule=step_schedule,
                            eta_schedule=eta_schedule,
                            inner_steps=args.inner_steps,
                        ).clamp(0, 1)

                        ps = psnr(out01.to(gt01.device), gt01)
                        ms = torch.tensor(float('nan'), device=device)
                        lp = patch_lpips_vgg(
                            lpips_fn,
                            out01.to(gt01.device),
                            gt01,
                            patch=int(getattr(args, "reward_lpips_patch", 224)),
                            stride=int(getattr(args, "reward_lpips_stride", 224)),
                        )
                        r = ps - float(args.lpips_weight) * lp

                        ci = torch.tensor(float('nan'), device=device)
                        qa = torch.tensor(float('nan'), device=device)
                        nv = torch.tensor(float('nan'), device=device)
                        ds = torch.tensor(float('nan'), device=device)
                        pc = torch.tensor(float('nan'), device=device)

                    r_runs.append(r)
                    ps_runs.append(ps)
                    ms_runs.append(ms)
                    lp_runs.append(lp)
                    clip_runs.append(ci)
                    qalign_runs.append(qa)
                    nrview_runs.append(nv)
                    dist_runs.append(ds)
                    perc_runs.append(pc)

                rewards[b, g] = torch.stack(r_runs).mean()
                psnrs[b, g] = torch.stack(ps_runs).mean()
                msssims[b, g] = torch.stack(ms_runs).mean()
                lpipss[b, g] = torch.stack(lp_runs).mean()
                clipiqas[b, g] = torch.stack(clip_runs).mean()
                qaligns[b, g] = torch.stack(qalign_runs).mean()
                nrviews[b, g] = torch.stack(nrview_runs).mean()
                dist_scores[b, g] = torch.stack(dist_runs).mean()
                perc_scores[b, g] = torch.stack(perc_runs).mean()
        # ================================
        # 1.6) GRPO advantage (group-relative) per image
        # ================================
        adv = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, unbiased=False, keepdim=True) + 1e-8)
        adv = adv.reshape(-1).detach()  # [B*G]

        rewards_flat = rewards.reshape(-1)
        psnrs_flat = psnrs.reshape(-1)
        msssims_flat = msssims.reshape(-1)
        lpipss_flat = lpipss.reshape(-1)

        clipiqas_flat = clipiqas.reshape(-1)
        qaligns_flat = qaligns.reshape(-1)
        nrviews_flat = nrviews.reshape(-1)
        dist_scores_flat = dist_scores.reshape(-1)
        perc_scores_flat = perc_scores.reshape(-1)

        # ================================
        # 2) PPO/GRPO update for multiple epochs on SAME rollout batch
        # ================================
        ratio_stats = None
        loss_hist = []

        # Use a fixed kl_beta for all epochs on the same rollout batch.
        kl_beta_cur = float(args.kl_beta)

        for ue in range(args.update_epochs):
            logp = policy.log_prob(phi)                 # [B*G] under current policy
            ratio = torch.exp(logp - logp_old)          # [B*G] new / old

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * adv
            obj = torch.min(surr1, surr2).mean()

            kl_ref = policy.kl_to(ref_alpha, ref_beta)
            loss = -(obj - kl_beta_cur * kl_ref)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()

            loss_hist.append(float(loss.detach().item()))

            # collect ratio stats from the last epoch (or you can keep min/max across epochs)
            if ue == args.update_epochs - 1:
                with torch.no_grad():
                    ratio_stats = (ratio.min().item(), ratio.mean().item(), ratio.max().item())

        # Recompute KL after the last optimizer step for logging/adaptation.
        with torch.no_grad():
            kl_val = float(policy.kl_to(ref_alpha, ref_beta).item())

        # === Adaptive KL control (PPO-style) ===
        # If KL(policy || ref) drifts above target_kl, increase kl_beta for the *next* iteration.
        kl_beta_next = kl_beta_cur
        if getattr(args, "kl_beta_adapt", False) and (it >= int(getattr(args, "kl_warmup_iters", 0))):
            target_kl = float(getattr(args, "target_kl", 0.0))
            if target_kl > 0:
                tol = float(getattr(args, "kl_tolerance", 0.0))
                hi = target_kl * (1.0 + tol)
                lo = target_kl * max(0.0, (1.0 - tol))

                if kl_val > hi:
                    kl_beta_next = kl_beta_cur * float(getattr(args, "kl_beta_up", 1.0))
                elif (float(getattr(args, "kl_beta_down", 1.0)) > 1.0) and (kl_val < lo):
                    kl_beta_next = kl_beta_cur / float(getattr(args, "kl_beta_down", 1.0))

                kl_beta_min = float(getattr(args, "kl_beta_min", 0.0))
                kl_beta_max = float(getattr(args, "kl_beta_max", float("inf")))
                kl_beta_next = float(min(max(kl_beta_next, kl_beta_min), kl_beta_max))

        # write back (used from next iteration / stored in ckpt)
        args.kl_beta = kl_beta_next

        R_mean_val = float(rewards_flat.mean().item())
        PSNR_val = float(psnrs_flat.mean().item())
        MSSSIM_val = float(_nanmean(msssims_flat).item())
        LPIPS_val = float(lpipss_flat.mean().item())

        # Modern IQA metrics (may be NaN if disabled)
        CLIP_IQA_val = float(_nanmean(clipiqas_flat).item())
        QALIGN_val = float(_nanmean(qaligns_flat).item())
        NRVIEWS_val = float(_nanmean(nrviews_flat).item())
        dist_score_val = float(_nanmean(dist_scores_flat).item())
        perc_score_val = float(_nanmean(perc_scores_flat).item())

        loss_val = float(sum(loss_hist) / max(1, len(loss_hist)))

        history["it"].append(it)
        history["R_mean"].append(R_mean_val)
        history["PSNR"].append(PSNR_val)
        history["LPIPS"].append(LPIPS_val)
        history["loss"].append(loss_val)

        # ---- NEW: write per-iter log ----
        row = {
            "it": int(it),
            "R_mean": float(rewards_flat.mean().item()),
            "PSNR": float(psnrs_flat.mean().item()),
            "MS_SSIM": float(MSSSIM_val),
            "LPIPS": float(lpipss_flat.mean().item()),
            "CLIP_IQA": float(CLIP_IQA_val),
            "QALIGN": float(QALIGN_val),
            "NRVIEWS": float(NRVIEWS_val),
            "distortion_score": float(dist_score_val),
            "perception_score": float(perc_score_val),
            "R_max": float(rewards_flat.max().item()),
            "KL": float(kl_val),
            "kl_beta": float(kl_beta_cur),
            "kl_beta_next": float(args.kl_beta),
            "target_kl": float(getattr(args, "target_kl", 0.0)),
            "loss": float(loss_val),
            "ratio_min": float(ratio_stats[0]),
            "ratio_mean": float(ratio_stats[1]),
            "ratio_max": float(ratio_stats[2]),
        }
        log_writer.writerow(row)
        log_f.flush()

        if wandb_run is not None:
            wandb.log(row)

        # logging
        pbar.set_postfix({
            "R_mean": float(rewards_flat.mean().item()),
            "PSNR": float(psnrs_flat.mean().item()),
            "MS-SSIM": float(MSSSIM_val),
            "LPIPS": float(lpipss_flat.mean().item()),
            "CLIP": float(CLIP_IQA_val),
            "QAlign": float(QALIGN_val),
            "NRViews": float(NRVIEWS_val),
            "R_max": float(rewards_flat.max().item()),
            "KL": float(kl_val),
            "kl_beta": float(kl_beta_cur),
            "kl_beta_next": float(args.kl_beta),
            "target_kl": float(getattr(args, "target_kl", 0.0)),
            "loss": float(loss_val),
            "ratio(min/mean/max)": f"{ratio_stats[0]:.3f}/{ratio_stats[1]:.3f}/{ratio_stats[2]:.3f}",
        })

        # ==============================
        # 3) Validation (optional)
        # ==============================
        if val_img_list is not None and int(getattr(args, "val_every", 0)) > 0:
            do_val = (it % int(args.val_every) == 0) or (it == args.iters - 1)
            if do_val:
                was_training = bool(getattr(policy, "training", False))
                try:
                    idx = 0
                    # Build operator per-image for motion deblur (kernel depends on seed)
                    if args.task == 'deblur_motion':
                        from functions.motionblur.motionblur import Kernel
                        if args.operator_imp == 'FFT':
                            from functions.fft_operators import Deblurring_fft
                        else:
                            raise ValueError("set operator_imp = FFT")

                        np.random.seed(seed=idx * 10)  # reproducibility per image
                        kernel = torch.from_numpy(
                            Kernel(size=(args.deg_scale, args.deg_scale), intensity=0.5).kernelMatrix
                        )
                        A_funcs_i = Deblurring_fft(kernel / kernel.sum(), 3, args.img_size, solver.transformer.device)
                    else:
                        A_funcs_i = A_funcs
                    idx += 1

                    policy.eval()
                    val_out = run_validation(
                        it=int(it),
                        policy=policy,
                        solver=solver,
                        A_funcs=A_funcs_i,
                        val_img_list=val_img_list,
                        prompts=prompts,
                        prompt_embs=prompt_embs_val,
                        null_embs=null_embs_val,
                        tf=tf,
                        bounds=bounds,
                        args=args,
                        device=device,
                        rewarder=rewarder,
                        lpips_fn=lpips_fn,
                        seed_sampling_fn=_seed_sampling,
                        max_images=int(getattr(args, "val_max_images", -1)),
                    )
                finally:
                    if was_training:
                        policy.train()

                # ---- save validation metrics ----
                if val_log_writer is not None:
                    val_row = {"it": int(it), **val_out}
                    val_log_writer.writerow(val_row)
                    if val_log_f is not None:
                        val_log_f.flush()

                # ---- store in history (for npz/plots) ----
                if "val_it" in history:
                    history["val_it"].append(int(it))
                    history["val_R_mean"].append(float(val_out.get("val_R_mean", float('nan'))))
                    history["val_PSNR"].append(float(val_out.get("val_PSNR", float('nan'))))
                    history["val_SSIM"].append(float(val_out.get("val_SSIM", float('nan'))))
                    history["val_LPIPS"].append(float(val_out.get("val_LPIPS", float('nan'))))
                    history["val_CLIP_IQA"].append(float(val_out.get("val_CLIP_IQA", float('nan'))))
                    history["val_QALIGN"].append(float(val_out.get("val_QALIGN", float('nan'))))
                    history["val_distortion_score"].append(float(val_out.get("val_distortion_score", float('nan'))))
                    history["val_perception_score"].append(float(val_out.get("val_perception_score", float('nan'))))
                    history["val_num_images"].append(int(val_out.get("val_num_images", 0)))

                # ---- wandb ----
                if wandb_run is not None:
                    wandb.log({
                        "it": int(it),
                        "val/R_mean": float(val_out.get("val_R_mean", float('nan'))),
                        "val/PSNR": float(val_out.get("val_PSNR", float('nan'))),
                        "val/SSIM": float(val_out.get("val_SSIM", float('nan'))),
                        "val/MS_SSIM": float(val_out.get("val_MS_SSIM", float('nan'))),
                        "val/LPIPS": float(val_out.get("val_LPIPS", float('nan'))),
                        "val/CLIP_IQA": float(val_out.get("val_CLIP_IQA", float('nan'))),
                        "val/QALIGN": float(val_out.get("val_QALIGN", float('nan'))),
                        "val/NRVIEWS": float(val_out.get("val_NRVIEWS", float('nan'))),
                        "val/distortion_score": float(val_out.get("val_distortion_score", float('nan'))),
                        "val/perception_score": float(val_out.get("val_perception_score", float('nan'))),
                        "val/num_images": int(val_out.get("val_num_images", 0)),
                    })

        if args.ckpt_every > 0 and it % args.ckpt_every == 0:
            save_ckpt(ckpt_dir, it, policy, optim, ref_alpha, ref_beta, bounds, args,
                      history=history, wandb_id=(wandb_run.id if wandb_run is not None else None), init_mu0=init_mu0)

    log_f.close()
    print(f"[Saved] iter log -> {log_path}")

    if val_log_f is not None:
        val_log_f.close()
        print(f"[Saved] val iter log -> {val_log_path}")

    if wandb_run is not None:
        wandb.finish()
    
    # ---- save metrics plots ----
    save_metrics_and_plots(args.workdir, history)

    # ---- plot schedule overlays (init vs final) ----
    with torch.no_grad():
        cfg_init, step_init, eta_init = phi_to_schedules(init_mu0, args.degree, args.NFE, bounds, device=device)
        cfg_final, step_final, eta_final = policy_to_schedules_mean(policy, args.degree, args.NFE, bounds, device=device)

    plot_schedule_overlay(args.workdir, "cfg", cfg_init, cfg_final)
    plot_schedule_overlay(args.workdir, "step", step_init, step_final)
    plot_schedule_overlay_eta(args.workdir, "eta", eta_init, eta_final)
    print(f"[Saved plots] metrics + schedules -> {args.workdir}")

    # save trained mean coeffs
    save_ckpt(ckpt_dir, args.iters - 1, policy, optim, ref_alpha, ref_beta, bounds, args,
              history=history, wandb_id=(wandb_run.id if wandb_run is not None else None), init_mu0=init_mu0, tag="_final")

if __name__ == "__main__":
    main()
