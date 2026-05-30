<div align="center">

# TriPS: Triadic Dynamics Aware Diffusion Posterior Sampling for Inverse Problems

### Optimizing Guidance and Stochasticity Schedules &nbsp;·&nbsp; ICML 2026

**Junseo Bang**<sup>1\*</sup> · **Dong Ju Mun**<sup>1\*</sup> · **Hoigi Seo**<sup>1</sup> · **Seongmin Hong**<sup>2</sup> · **Se Young Chun**<sup>1,2,3</sup>

<sup>1</sup>ECE · <sup>2</sup>INMC · <sup>3</sup>IPAI &amp; AIIS, Seoul National University &nbsp; (<sup>\*</sup>equal contribution)

[![Paper](https://img.shields.io/badge/Paper-PMLR%20306-b31b1b.svg)](https://github.com/mundongju/TriPS)
[![Code](https://img.shields.io/badge/GitHub-TriPS-181717.svg?logo=github)](https://github.com/mundongju/TriPS)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## TL;DR

Diffusion posterior sampling for inverse problems is governed by **three coupled components** — **data consistency (DC)**, **classifier-free guidance (CFG)**, and **stochasticity (STO)**. Their *schedules* are usually fixed by heuristics. **TriPS** treats sampling as a time-varying control problem and optimizes these schedules along a **triadic trend: DC ↓, CFG ↑, STO ↓**, via two routes:

| | What | How |
|---|---|---|
| **TriPS-T** | training-free **template** schedules | grid search over `linear / logarithm / exponential` priors → [`TriPS_T/`](TriPS_T) |
| **TriPS-G** | learned **GRPO** schedules | RL fine-tuning of the schedule, initialized from TriPS-T → [`TriPS_G/`](TriPS_G) |

| Component | Code key | Trend |
|---|---|---|
| Data consistency (gradient step) | `step_scale` | ↓ |
| Classifier-free guidance | `cfg` | ↑ |
| Stochasticity (noise injection) | `eta` | ↓ |

Prior model: **Stable Diffusion 3.5-Medium**, `NFE = 28`.

---

## Repository layout

```
TriPS/
├── inference.py            # ⭐ unified inference: TriPS-T (template) or TriPS-G (GRPO ckpt)
├── run_inference.sh        # one-command demo on ./demo_images
├── run_eval_patch.sh       # patch-FID / patch-KID
├── eval.py                 # full-reference metrics (PSNR / SSIM / LPIPS)
├── compute_patch_FID.py · compute_patch_KID.py
├── sd3_sampler_total.py    # TriPS-T sampler (template schedules)
├── sd3_sampler_ours_test.py# TriPS-G inference sampler (explicit schedules)
├── grpo_schedule.py · custom_util.py · util.py · cores/ · functions/ · motionblur/   # shared
├── demo_images/            # FFHQ (faces) + DIV2K (scenes) + prompts
├── TriPS.yaml
│
├── TriPS_T/                # TriPS-T grid search → Excel  (find the template schedule)
│   ├── solve.py · run_search.sh · inp_masks/ · DIV2K_prompts.txt
│
└── TriPS_G/                # TriPS-G GRPO training  (+ export the reference policy)
    ├── build_init_schedules.py   # TriPS-T schedules → init_load_file_fin/*.npz
    ├── train_grpo_schedule_w_val.py · sd3_sampler_ours.py · iqa_reward.py · run_train.sh
    └── init_load_file_fin/ · Datasets/ · exp/
```

Shared modules live at the **root**; the runner scripts add the root to `PYTHONPATH` automatically.

---

## Setup

```bash
git clone https://github.com/mundongju/TriPS.git
cd TriPS
conda env create -f TriPS.yaml -n TriPS
conda activate TriPS
```

SD3.5-M is fetched from the Hugging Face Hub on first run (`huggingface-cli login` may be needed). One 24 GB GPU suffices with `--efficient_memory`.

---

## 🚀 Inference (start here)

Run a **fixed, already-found schedule** on the demo images. Tasks: `0`=SR×8, `1`=Gaussian deblur, `2`=motion deblur.

```bash
# TriPS-T  (training-free template schedule)
bash run_inference.sh TriPS-T 0

# TriPS-G  (learned schedule; needs a trained checkpoint)
GRPO_CKPT=/path/to/grpo_schedule_ckpt_sr_bicubic_itXXXX.pt bash run_inference.sh TriPS-G 0
```

Or call `inference.py` directly:

```bash
python inference.py --method TriPS-T --dataset DIV2K --task sr_bicubic \
    --img_path demo_images/DIV2K --prompt_file demo_images/DIV2K/DIV2K_prompts_demo.txt \
    --workdir results/TriPS-T/srx8 --efficient_memory

python inference.py --method TriPS-G --task sr_bicubic --operator_imp SVD --deg_scale 8 \
    --img_path demo_images/DIV2K --prompt_file demo_images/DIV2K/DIV2K_prompts_demo.txt \
    --grpo_ckpt /path/to/ckpt.pt --workdir results/TriPS-G/srx8 --efficient_memory
```

- **TriPS-T**: `--dataset {DIV2K,FFHQ} --task {sr_bicubic,deblur_gauss,deblur_motion}` selects the paper's confirmed template schedule (override with `--function_dc/_cfg/_sto`, `--step_scale`, `--step_scale_2`).
- **TriPS-G**: `--grpo_ckpt` is a checkpoint from TriPS-G training; its `cfg/step/eta` curves are read from the policy.

Results: `results/<method>/<task>/{recon,label,input1,...}` + `eval_results.txt`.

---

## Reproduce the schedules

**TriPS-T (find template schedules → Excel).** Grid-searches `linear/log/exp` priors and logs an `eval_score` per run:
```bash
cd TriPS_T && bash run_search.sh 0      # see TriPS_T/README.md
```

**TriPS-G (train the learned schedule).** Initialized from the TriPS-T reference policy:
```bash
cd TriPS_G
python build_init_schedules.py --verify   # TriPS-T schedules → init_load_file_fin/*.npz
bash run_train.sh 1                        # see TriPS_G/README.md
```

---

## Evaluation

```bash
bash run_eval_patch.sh fid results/TriPS-T/sr_bicubic/label results/TriPS-T/sr_bicubic/recon 256
bash run_eval_patch.sh kid results/TriPS-T/sr_bicubic/label results/TriPS-T/sr_bicubic/recon 192
```

Paper setup: SD3.5-M, 28 NFE, noise `σ_n=0.03`, 1000 FFHQ / 800 DIV2K; SR ×8/×12 (bicubic), motion deblur (61×61, intensity 0.5), Gaussian deblur (σ=3.0). Metrics: PSNR / SSIM / FID / LPIPS.

---

## Citation

```bibtex
@inproceedings{bang2026trips,
  title     = {Triadic Dynamics Aware Diffusion Posterior Sampling for Inverse Problems:
               Optimizing Guidance and Stochasticity Schedules},
  author    = {Bang, Junseo and Mun, Dong Ju and Seo, Hoigi and Hong, Seongmin and Chun, Se Young},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}, series = {PMLR}, volume = {306}
}
```

## Acknowledgements & License

Builds on [FlowDPS](https://github.com/FlowDPS-Inverse/FlowDPS), [Stable Diffusion 3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) (🤗 `diffusers`), and [motionblur](https://github.com/LeviBorodenko/motionblur). Code released under the [MIT License](LICENSE); SD3.5 weights follow the Stability AI Community License.
