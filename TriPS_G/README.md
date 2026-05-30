# TriPS-G — GRPO schedule training

Refines the TriPS-T template schedule into a flexible learned schedule with **GRPO**
(Group Relative Policy Optimization), using a hybrid IQA reward (PSNR + LPIPS, optionally
CLIP-IQA / Q-Align) and KL-anchoring to the TriPS-T reference policy.

> `run_train.sh` adds the repo root to `PYTHONPATH` and `cd`s here, so `grpo_schedule`,
> `util`, `cores`, `functions` (at the root) and the local `sd3_sampler_ours` / `iqa_reward` import fine.

## Dataset
GRPO training for TriPS-G uses images that are **out-of-distribution** w.r.t. the TriPS-T
grid search — i.e. images **not** used in the template search — so the learned schedule does
not overfit the calibration set. Place them under `./Datasets/` in four subfolders:

```
Datasets/
├── FFHQ_GRPO_train_100/   # FFHQ 01000.png ... 01099.png   (100 train images)
├── FFHQ_GRPO_val_10/      # FFHQ 01100.png ... 01109.png   (10 val images)
├── DIV2K_GRPO_train_100/  # DIV2K 0801.png ... 0900.png    (100 train images)
└── DIV2K_GRPO_val_10/     # DIV2K 0901.png ... 0910.png    (10 val images)
```

- **FFHQ** — from the official [FFHQ repository](https://github.com/NVlabs/ffhq-dataset):
  `01000.png`–`01099.png` → `FFHQ_GRPO_train_100/`, and `01100.png`–`01109.png` → `FFHQ_GRPO_val_10/`.
- **DIV2K** — from the official [DIV2K page](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (`DIV2K_train_HR`):
  `0801.png`–`0900.png` → `DIV2K_GRPO_train_100/`, and `0901.png`–`0910.png` → `DIV2K_GRPO_val_10/`.

The train/val splits are disjoint, and both are disjoint from the TriPS-T search images
(`FFHQ 00000–00999`, `DIV2K 0001–0800`).

## 1) Build the reference policy (TriPS-T → npz)

Six files `{DIV2K,FFHQ} × {sr_bicubic, Gauss_deblur, motion_deblur}` ship in
`init_load_file_fin/`; each holds `sigma`, `eta`, `cfg`, `step_scale` (length `NFE`).
Regenerate deterministically from the TriPS-T templates:

```bash
python build_init_schedules.py --verify     # check shipped npz == templates
python build_init_schedules.py --force       # regenerate all 6
```

## 2) Train

```bash
bash run_train.sh 0   # gaussian deblur
bash run_train.sh 1   # SR x8
bash run_train.sh 2   # motion deblur
```

Key flags (full list in `train_grpo_schedule_w_val.py`): `--init_load_file` (reference policy),
`--group_size`, `--img_batch`, `--reward_runs`, `--iters`, `--lr`, `--kl_beta`, `--clip_eps`,
the `--reward_*` family, `--use_wandb` / `--wandb_mode`. Checkpoints → `<workdir>/ckpts/*.pt`.

## 3) Inference

Use the root [`inference.py`](../inference.py) with the trained checkpoint:

```bash
GRPO_CKPT=<workdir>/ckpts/grpo_schedule_ckpt_sr_bicubic_itXXXX.pt \
    bash ../run_inference.sh TriPS-G 0
```
