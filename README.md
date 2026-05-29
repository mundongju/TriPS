<div align="center">

# TriPS: Triadic Dynamics Aware Diffusion Posterior Sampling for Inverse Problems

### Optimizing Guidance and Stochasticity Schedules

**Junseo Bang**<sup>1\*</sup> &nbsp;·&nbsp; **Dong Ju Mun**<sup>1\*</sup> &nbsp;·&nbsp; **Hoigi Seo**<sup>1</sup> &nbsp;·&nbsp; **Seongmin Hong**<sup>2</sup> &nbsp;·&nbsp; **Se Young Chun**<sup>1,2,3</sup>

<sup>1</sup>Dept. of Electrical and Computer Engineering, &nbsp; <sup>2</sup>INMC, &nbsp; <sup>3</sup>IPAI &amp; AIIS &nbsp;—&nbsp; Seoul National University, Republic of Korea
<br><sup>\*</sup>Equal contribution &nbsp;·&nbsp; Correspondence: `sychun@snu.ac.kr`

**International Conference on Machine Learning (ICML) 2026**

[![Paper](https://img.shields.io/badge/Paper-PMLR%20306-b31b1b.svg)](https://github.com/mundongju/TriPS)
[![Project Page](https://img.shields.io/badge/Project-Page-1f8b4c.svg)](https://github.com/mundongju/TriPS)
[![Code](https://img.shields.io/badge/GitHub-TriPS-181717.svg?logo=github)](https://github.com/mundongju/TriPS)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#-license)

</div>

---

## 📌 Abstract

> Generative posterior sampling using diffusion models has emerged as a dominant paradigm for solving inverse problems in imaging, which usually consists of **three main components: data-consistency (DC) guidance, classifier-free guidance (CFG), and stochasticity.**

❗️ While prior arts have focused on *how to develop* each (or all) of these components, **less attention has been given to *how to schedule* them**, leading to heuristically fixed or partially adjusted, suboptimal schedules.

❓ In this work we argue that the **interactions among all three components — in terms of scheduling — are crucial** for significantly improved performance. Our analysis shows that *aggressive CFG early in sampling conflicts with DC guidance*, while *stochasticity brings the trajectory back to higher-probability regions*.

👍 Based on these findings, we propose **Triadic Dynamics Aware Posterior Sampling (TriPS)**, which reformulates posterior sampling as a **time-varying control problem** and optimizes schedules following a **triadic trend: decreasing DC and stochasticity scales alongside an increasing CFG scale.** TriPS realizes this through two complementary strategies:

| | Strategy | Folder |
|---|---|---|
| **TriPS-T** | **T**emplate-based search over functional priors (`linear` / `logarithm` / `exponential`) for reliable baseline schedules — *training-free*. | [`TriPS_T/`](TriPS_T) |
| **TriPS-G** | **G**RPO-based reinforcement learning (Group Relative Policy Optimization) for more flexible temporal curves — *learned*. | [`TriPS_G/`](TriPS_G) |

Experiments on Stable Diffusion 3.5 show that TriPS outperforms state-of-the-art baselines in both **data fidelity and perceptual realism**.

---

## 🔑 The Triad

TriPS schedules three coupled components over the `NFE = 28` reverse-time steps. The TriPS-T final schedules (also used to initialize TriPS-G) follow the triadic trend below:

| Component | Symbol | Code key | Trend |
|---|---|---|---|
| Data consistency (gradient step size) | DC | `step_scale` | **decreasing** ↓ |
| Classifier-free guidance | CFG | `cfg` | **increasing** ↑ |
| Stochasticity (noise injection) | STO | `eta` | **decreasing** ↓ |
| Noise level of the prior (fixed by SD3.5) | σ | `sigma` | decreasing ↓ |

These four length-`NFE` arrays are serialized into `TriPS_G/init_load_file_fin/*.npz` and consumed by GRPO as its **reference policy**.

---

## 🗂️ Repository structure

The shared building blocks live at the repository **root**, and the two methods live in their own folders:

```
TriPS/
├── README.md                      # this file
├── TriPS.yaml                     # conda environment (Python 3.10, torch 2.4.1, diffusers 0.36)
│
├── util.py                        # ── shared utilities (set_seed, get_img_list, process_text)
├── cores/                         # ── shared diffusion cores
│   ├── scheduler.py               #     VP/VE/EDM/TrigFlow schedulers + PF-ODE
│   ├── mcmc.py                    #     Langevin / MCMC sampling
│   └── trajectory.py
├── functions/                     # ── shared forward operators (A, A^T, A^+)
│   ├── svd_operators.py           #     SR / inpainting / colorization / deblur (SVD)
│   ├── fft_operators.py           #     FFT-based (super-res, deblur)
│   ├── ddpg_scheme.py             #     data-consistency helpers
│   ├── phase_retrieval_operator*.py
│   ├── ckpt_util.py
│   └── motionblur/                #     motion-blur kernel generator
├── motionblur/                    # ── (mirror used by some operators)
│
├── demo_images/                   # ── tiny demo set for quick runs
│   ├── FFHQ/                       #     3 face images
│   └── DIV2K/                      #     3 scenes + DIV2K_prompts_demo.txt
│
├── TriPS_T/                       # ===== TriPS-T : training-free template solver =====
│   ├── solve.py                   #     entry point
│   ├── sd3_sampler_total.py       #     SD3.5 sampler w/ triadic template schedules
│   ├── custom_util.py · eval.py
│   ├── compute_patch_FID_final.py · compute_patch_KID_final.py · pFID.sh
│   ├── inp_masks/ · DIV2K_prompts.txt
│   ├── run_demo_TriPS_T.sh        #     full grid runner (original)
│   └── run_demo_TriPS_T_demo.sh   #     ⭐ demo runner (uses ../demo_images)
│
└── TriPS_G/                       # ===== TriPS-G : GRPO schedule optimization =====
    ├── train_grpo_schedule_w_val.py   # GRPO training (+ validation)
    ├── grpo_schedule.py · iqa_reward.py
    ├── solve_ours.py              #     inference with a trained schedule
    ├── sd3_sampler_ours.py · sd3_sampler_ours_test.py · eval.py
    ├── build_init_schedules.py    #     ⭐ export TriPS-T schedules -> init_load_file_fin/*.npz
    ├── init_load_file_fin/        #     6 reference-policy .npz (sigma/eta/cfg/step_scale)
    ├── Datasets/ · exp/inp_masks/ · DIV2K_prompts*.txt
    ├── run_demo_TriPS_G_train.sh · run_demo_TriPS_G_test.sh
    └── run_demo_TriPS_G_demo.sh   #     ⭐ demo runner (uses ../demo_images)
```

> **Why this layout?** `util.py`, `cores/`, `functions/`, and `motionblur/` are byte-for-byte identical for both methods, so they are kept once at the root and shared. Each runner adds the repo root to `PYTHONPATH` automatically (see the bootstrap header in every `*.sh`), so the imports `from util import ...`, `from cores... import ...`, `from functions... import ...` resolve from the root, while task-specific modules (`eval.py`, `custom_util.py`, `solve*.py`, …) resolve from each method folder.

---

## ⚙️ Environment Setup

```bash
git clone https://github.com/mundongju/TriPS.git
cd TriPS

# create the conda environment (Python 3.10, CUDA 11.8/12.1 wheels)
conda env create -f TriPS.yaml
conda activate TriPS
```

The prior model is **Stable Diffusion 3.5 Medium**, downloaded automatically from the Hugging Face Hub on first run (you may need `huggingface-cli login` for gated access). A single 24 GB GPU is sufficient when `--efficient_memory` is enabled (the text encoder pre-computes embeddings and is offloaded).

---

## 🚀 Quick Start (demo)

Both methods can be exercised end-to-end on the bundled `demo_images/`.

### TriPS-T (training-free)

```bash
cd TriPS_T
bash run_demo_TriPS_T_demo.sh 0   # super-resolution x8 (bicubic), FFHQ
bash run_demo_TriPS_T_demo.sh 1   # gaussian deblur,               FFHQ
bash run_demo_TriPS_T_demo.sh 2   # motion deblur,                 DIV2K
# outputs -> TriPS_T/workdir_TriPS_T_demo/<task>/
```

### TriPS-G (GRPO)

```bash
cd TriPS_G

# (a) short GRPO training demo, initialized from the TriPS-T reference policy
bash run_demo_TriPS_G_demo.sh 0          # 0=sr_bicubic | 1=deblur_gauss | 2=deblur_motion
# outputs + checkpoints -> TriPS_G/workdir_TriPS_G_demo_<task>/

# (b) inference with a trained schedule checkpoint
GRPO_CKPT=workdir_TriPS_G_demo_sr_bicubic/ckpts/grpo_schedule_ckpt_sr_bicubic_it0001.pt \
    bash run_demo_TriPS_G_demo.sh 0 test
```

---

## 🧩 TriPS-T — Training-free template solver

TriPS-T builds each schedule by interpolating between two endpoints with a chosen
*functional prior* (`--function_dc` / `--function_cfg` / `--function_sto` ∈ `{linear, logarithm, exponential}`).

```bash
cd TriPS_T
python solve.py \
    --img_size 768 \
    --img_path ../demo_images/FFHQ \
    --prompt "a high quality photo of a face" \
    --method TriPS_T --task sr_bicubic --operator_imp SVD --deg_scale 8 \
    --noise_std 0.03 --cfg_scale 2.0 --seed 42 --NFE 28 \
    --step_scale 250 --step_scale_2 40 --inner_steps 6 --stochasticity_weight 1.0 \
    --function_dc linear --function_cfg logarithm --function_sto logarithm \
    --workdir workdir_TriPS_T_demo/SRx8 --efficient_memory
```

**Tasks** (`--task`): `sr_bicubic`, `deblur_gauss`, `deblur_motion`, `inpainting`, … &nbsp;|&nbsp;
**Operator backend** (`--operator_imp`): `SVD` or `FFT`. For SR, `--deg_scale` is the downscale factor; for deblurring, the kernel size.

The full reproduction grid (all functional priors, full FFHQ/DIV2K sets) is in
[`run_demo_TriPS_T.sh`](TriPS_T/run_demo_TriPS_T.sh).

---

## 🤖 TriPS-G — GRPO schedule optimization

### 1) Build the reference policy (TriPS-T → `.npz`)

TriPS-G starts from the **confirmed TriPS-T schedules**, stored as length-`NFE`
arrays (`sigma`, `eta`, `cfg`, `step_scale`) in `init_load_file_fin/*.npz`.
These files ship with the repo and can be (re)generated deterministically:

```bash
cd TriPS_G
python build_init_schedules.py --verify       # check shipped npz == template definitions
python build_init_schedules.py --force        # regenerate all 6 (DIV2K/FFHQ × 3 tasks)
```

`build_init_schedules.py` reproduces the exact template-shaping math used by the
TriPS-T sampler, so the exported schedules are consistent with what TriPS-T runs.

### 2) Train

```bash
cd TriPS_G
bash run_demo_TriPS_G_train.sh 1              # full config: 0=gauss | 1=sr | 2=motion
```

GRPO maximizes a hybrid IQA reward (perceptual e.g. **LPIPS** + distortion e.g. **PSNR**,
optionally CLIP-IQA / Q-Align) while staying close (KL) to the TriPS-T reference policy.
Checkpoints are written to `<workdir>/ckpts/`.

### 3) Inference

```bash
cd TriPS_G
python solve_ours.py \
    --img_size 768 --img_path ../demo_images/DIV2K \
    --prompt_file ../demo_images/DIV2K/DIV2K_prompts_demo.txt \
    --method flowdps_moon --task sr_bicubic --operator_imp SVD --deg_scale 8 \
    --noise_std 0.03 --cfg_scale 2.0 --seed 42 --NFE 28 --inner_steps 6 \
    --grpo_ckpt /path/to/grpo_schedule_ckpt_sr_bicubic_itXXXX.pt \
    --workdir workdir_TriPS_G_test/sr_bicubic --efficient_memory
```

---

## 📊 Evaluation

Patch-FID / patch-KID scripts are provided for both methods:

```bash
bash TriPS_T/pFID.sh          # patch-FID / patch-KID for TriPS-T outputs
bash TriPS_G/pFID_eval.sh     # patch-FID for TriPS-G outputs
```

Reported settings: Stable Diffusion 3.5-M prior, **28 NFE**, Gaussian measurement
noise `σ_n = 0.03`, evaluated over **1,000 FFHQ** and **800 DIV2K** samples for
super-resolution ×8/×12 (bicubic), motion deblurring (61×61 kernel, intensity 0.5),
and Gaussian deblurring (σ = 3.0). Metrics: PSNR, SSIM, FID, LPIPS.

---

## 📖 Citation

If you find TriPS useful, please cite:

```bibtex
@inproceedings{bang2026trips,
  title     = {Triadic Dynamics Aware Diffusion Posterior Sampling for Inverse Problems:
               Optimizing Guidance and Stochasticity Schedules},
  author    = {Bang, Junseo and Mun, Dong Ju and Seo, Hoigi and Hong, Seongmin and Chun, Se Young},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
  series    = {PMLR},
  volume    = {306}
}
```

---

## 🙏 Acknowledgements

This codebase builds on the inverse-problem / flow-matching ecosystem, in particular
[FlowDPS](https://github.com/FlowDPS-Inverse/FlowDPS), the
[Stable Diffusion 3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) prior via
🤗 `diffusers`, and the [motionblur](https://github.com/LeviBorodenko/motionblur) kernel generator.
We thank the authors of these projects.

## 📄 License

Released under the MIT License (see `LICENSE`). Stable Diffusion 3.5 weights are subject to the
Stability AI Community License.
