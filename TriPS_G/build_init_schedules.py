#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_init_schedules.py
=======================

Export the **TriPS-T final (confirmed) template schedules** into the
``init_load_file_fin/`` ``.npz`` files that **TriPS-G** (GRPO) loads as its
*reference policy* (a.k.a. the schedule it is initialised from and KL-regularised
towards). See ``train_grpo_schedule_w_val.py::build_ref_schedules`` which reads the
``cfg`` / ``step_scale`` / ``eta`` keys, and the training launcher
``run_demo_TriPS_G_train.sh`` (flag ``--init_load_file``).

Why this file exists
--------------------
The paper "Triadic Dynamics Aware Diffusion Posterior Sampling for Inverse
Problems" (ICML 2026) proposes TriPS, which schedules a *triad* of components:

    * DC  (data-consistency)  -> ``step_scale``   : decreasing
    * CFG (classifier-free g.) -> ``cfg``          : increasing
    * STO (stochasticity)      -> ``eta``          : decreasing

TriPS-T finds reliable baseline schedules by a **template-based search over
functional priors** (``linear`` / ``logarithm`` / ``exponential`` shapes).
TriPS-G then refines them with **GRPO**. To bootstrap GRPO we serialise the
*confirmed* TriPS-T schedules per (dataset, task) into ``.npz`` files holding four
length-``NFE`` arrays: ``sigma``, ``eta``, ``cfg``, ``step_scale``.

This script regenerates exactly those files. The template-shaping functions and
their hyper-parameters are copied verbatim from the TriPS-T sampler
(``../TriPS_T/sd3_sampler_total.py``, the ``sample()`` loop and the
``lerp / phi_exp / phi_log / phi_exp_latedrop`` helpers) so the output is
bit-for-bit consistent with what the sampler actually used.

Safety
------
* By default existing ``.npz`` files are **NOT overwritten** (the repository ships
  the verified files). Use ``--force`` to regenerate them.
* ``--verify`` compares freshly built arrays against the shipped files and reports
  the maximum absolute difference per key (it writes nothing).

Usage
-----
    # Verify the shipped npz match the template definitions (writes nothing):
    python build_init_schedules.py --verify

    # (Re)generate only missing files into ./init_load_file_fin :
    python build_init_schedules.py

    # Force-regenerate every file:
    python build_init_schedules.py --force

    # Generate a single (dataset, task):
    python build_init_schedules.py --dataset DIV2K --task sr_bicubic --force

Only depends on numpy.
"""

import argparse
import os

import numpy as np

# Number of function evaluations (sampling steps). TriPS uses 28 throughout.
NFE = 28

# ---------------------------------------------------------------------------
# Canonical sigma schedule (noise level sigma_t along the reverse trajectory).
#
# This is the Stable-Diffusion-3.5 ``FlowMatchEulerDiscreteScheduler`` schedule
# obtained with ``scheduler.config.shift = 4.0`` and ``set_timesteps(NFE)``,
# i.e. ``sigmas = timesteps / num_train_timesteps`` exactly as computed inside
# ``sd3_sampler_total.py``. It is *identical* for every dataset/task (it depends
# only on the prior model, not on the inverse problem), so we embed the verified
# values here to keep this exporter dependency-free and reproducible.
# ---------------------------------------------------------------------------
SIGMA_SHIFT4_NFE28 = np.array([
    1.0,
    0.98738068342208862, 0.97410780191421509, 0.96012938022613525,
    0.94538754224777222, 0.92981797456741333, 0.91334903240203857,
    0.89590036869049072, 0.87738192081451416, 0.85769236087799072,
    0.83671671152114868, 0.81432479619979858, 0.79036831855773926,
    0.76467716693878174, 0.73705589771270752, 0.70727854967117310,
    0.67508238554000854, 0.64016026258468628, 0.60215061902999878,
    0.56062501668930054, 0.51507210731506348, 0.46487605571746826,
    0.40928882360458374, 0.34739264845848083, 0.27804881334304810,
    0.19982698559761047, 0.11090573668479919, 0.0089285718277096748,
], dtype=np.float64)
assert SIGMA_SHIFT4_NFE28.shape == (NFE,)


# ===========================================================================
# Template shaping functions -- copied verbatim from TriPS_T/sd3_sampler_total.py
# ===========================================================================
def lerp(a, b, t):
    """1) linear interpolation."""
    return a + (b - a) * t


def phi_exp(u, alpha=5.0):
    """2) exponential (normalized), convex / slow-start."""
    return (np.exp(alpha * u) - 1.0) / (np.exp(alpha) - 1.0)


def phi_exp_latedrop(u, alpha=8.0):
    """Stable late-drop variant of phi_exp using expm1 (used by sto='logarithm')."""
    u = np.asarray(u, dtype=np.float64)
    return np.expm1(alpha * u) / np.expm1(alpha)


def phi_log(u, alpha=9.0):
    """3) logarithmic (normalized), concave / fast-start."""
    return np.log1p(alpha * u) / np.log1p(alpha)


# ===========================================================================
# Per-component schedule builders. The string -> shape mapping reproduces the
# exact branches of the TriPS_T sampling loop (note that, by the sampler's own
# naming, dc='logarithm' uses phi_exp while cfg='logarithm' uses phi_log, etc.).
# ===========================================================================
def build_step_scale(function_dc, step_scale, step_scale_2):
    """DC (data-consistency) gradient-step size schedule: step_scale -> step_scale_2."""
    out = np.empty(NFE, dtype=np.float64)
    for i in range(NFE):
        u = i / (NFE - 1)
        if function_dc == "linear":
            out[i] = lerp(step_scale, step_scale_2, u)
        elif function_dc == "logarithm":
            out[i] = lerp(step_scale, step_scale_2, phi_exp(u, 6.0))
        elif function_dc == "exponential":
            out[i] = lerp(step_scale, step_scale_2, phi_log(u, 6.0))
        else:
            raise ValueError(f"Unknown function_dc='{function_dc}'")
    return out


def build_cfg(function_cfg, cfg_lo=1.0, cfg_hi=6.0):
    """CFG (classifier-free guidance) scale schedule: cfg_lo -> cfg_hi."""
    out = np.empty(NFE, dtype=np.float64)
    for i in range(NFE):
        u = i / (NFE - 1)
        if function_cfg == "linear":
            out[i] = lerp(cfg_lo, cfg_hi, u)
        elif function_cfg == "exponential":
            out[i] = lerp(cfg_lo, cfg_hi, phi_exp(u, 6.0))
        elif function_cfg == "logarithm":
            out[i] = lerp(cfg_lo, cfg_hi, phi_log(u, 6.0))
        else:
            raise ValueError(f"Unknown function_cfg='{function_cfg}'")
    return out


def build_eta(function_sto, stochasticity_weight=1.0):
    """STO (stochasticity) noise-injection schedule eta: 1 -> 0."""
    out = np.empty(NFE, dtype=np.float64)
    for i in range(NFE):
        u = i / (NFE - 1)
        if function_sto == "linear":
            alpha = lerp(1.0, 0.0, u)
        elif function_sto == "logarithm":
            t_exp = phi_exp_latedrop(u, 15.0)
            alpha = lerp(1.0, 0.0, t_exp) * stochasticity_weight
        elif function_sto == "exponential":
            t_log = phi_log(u, 50.0)
            alpha = lerp(1.0, 0.0, t_log)
        else:
            raise ValueError(f"Unknown function_sto='{function_sto}'")
        if alpha > 1.0:
            alpha = 1.0
        out[i] = alpha
    return out


# ===========================================================================
# TriPS-T final (confirmed) template configuration per (dataset, task).
#
# These are the schedules selected by the TriPS-T template search and reported in
# the paper; they reproduce the shipped init_load_file_fin/*.npz exactly.
#   filename : output file name (matches the names train_grpo_schedule expects)
#   dc/cfg/sto : template shape per triad component
#   step_scale / step_scale_2 : DC start / end values
#   stochasticity_weight : multiplier used by the sto='logarithm' branch
# ===========================================================================
CONFIGS = {
    ("DIV2K", "sr_bicubic"): dict(
        filename="DIV2K_sr_bicubic_sigma_eta_cfg_dc_ref_policy_fin.npz",
        dc="linear", cfg="logarithm", sto="logarithm",
        step_scale=300.0, step_scale_2=100.0, stochasticity_weight=1.0),
    ("DIV2K", "deblur_gauss"): dict(
        filename="DIV2K_Gauss_deblur_sigma_eta_cfg_dc_ref_policy_fin.npz",
        dc="linear", cfg="logarithm", sto="logarithm",
        step_scale=300.0, step_scale_2=200.0, stochasticity_weight=1.0),
    ("DIV2K", "deblur_motion"): dict(
        filename="DIV2K_motion_deblur_sigma_eta_cfg_dc_ref_policy_fin.npz",
        dc="linear", cfg="exponential", sto="linear",
        step_scale=350.0, step_scale_2=150.0, stochasticity_weight=1.0),
    ("FFHQ", "sr_bicubic"): dict(
        filename="FFHQ_sr_bicubic_sigma_eta_cfg_dc_ref_policy_fin.npz",
        dc="linear", cfg="logarithm", sto="logarithm",
        step_scale=250.0, step_scale_2=40.0, stochasticity_weight=1.0),
    ("FFHQ", "deblur_gauss"): dict(
        filename="FFHQ_Gauss_deblur_sigma_eta_cfg_dc_ref_policy_fin.npz",
        dc="logarithm", cfg="logarithm", sto="logarithm",
        step_scale=200.0, step_scale_2=100.0, stochasticity_weight=1.0),
    ("FFHQ", "deblur_motion"): dict(
        filename="FFHQ_motion_deblur_sigma_eta_cfg_dc_ref_policy_fin.npz",
        dc="linear", cfg="exponential", sto="linear",
        step_scale=350.0, step_scale_2=150.0, stochasticity_weight=1.0),
}


def build_arrays(cfg_entry):
    """Return the four (NFE,) float64 arrays for one config entry."""
    sigma = SIGMA_SHIFT4_NFE28.copy()
    eta = build_eta(cfg_entry["sto"], cfg_entry["stochasticity_weight"])
    cfg = build_cfg(cfg_entry["cfg"])
    step_scale = build_step_scale(
        cfg_entry["dc"], cfg_entry["step_scale"], cfg_entry["step_scale_2"])
    return dict(sigma=sigma, eta=eta, cfg=cfg, step_scale=step_scale)


def main():
    ap = argparse.ArgumentParser(
        description="Export TriPS-T final template schedules to init_load_file_fin/*.npz")
    ap.add_argument("--out_dir", default="init_load_file_fin",
                    help="output directory (default: init_load_file_fin)")
    ap.add_argument("--dataset", choices=["DIV2K", "FFHQ"], default=None,
                    help="restrict to one dataset (default: all)")
    ap.add_argument("--task",
                    choices=["sr_bicubic", "deblur_gauss", "deblur_motion"],
                    default=None, help="restrict to one task (default: all)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing .npz files")
    ap.add_argument("--verify", action="store_true",
                    help="compare against existing .npz and report max abs diff "
                         "(writes nothing)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    selected = [
        (k, v) for k, v in CONFIGS.items()
        if (args.dataset is None or k[0] == args.dataset)
        and (args.task is None or k[1] == args.task)
    ]

    for (dataset, task), entry in selected:
        path = os.path.join(args.out_dir, entry["filename"])
        arrays = build_arrays(entry)

        if args.verify:
            if not os.path.exists(path):
                print(f"[verify] MISSING  {path}")
                continue
            ref = np.load(path, allow_pickle=True)
            msgs = []
            for key in ("sigma", "eta", "cfg", "step_scale"):
                ref_v = ref[key].astype(np.float64)
                diff = float(np.max(np.abs(ref_v - arrays[key])))
                msgs.append(f"{key}:{diff:.3e}")
            print(f"[verify] {dataset:5s} {task:13s} max|diff|  " + "  ".join(msgs))
            continue

        if os.path.exists(path) and not args.force:
            print(f"[skip ] exists (use --force): {path}")
            continue

        np.savez(path, **arrays)
        print(f"[write] {dataset:5s} {task:13s} -> {path}")

    if not args.verify:
        print("\nDone. These .npz are consumed by train_grpo_schedule_w_val.py "
              "via --init_load_file (see run_demo_TriPS_G_train.sh).")


if __name__ == "__main__":
    main()
