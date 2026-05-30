# TriPS-G — GRPO schedule training

Refines the TriPS-T template schedule into a flexible learned schedule with **GRPO**
(Group Relative Policy Optimization), using a hybrid IQA reward (PSNR + LPIPS, optionally
CLIP-IQA / Q-Align) and KL-anchoring to the TriPS-T reference policy.

> `run_train.sh` adds the repo root to `PYTHONPATH` and `cd`s here, so `grpo_schedule`,
> `util`, `cores`, `functions` (at the root) and the local `sd3_sampler_ours` / `iqa_reward` import fine.

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

## Files

`build_init_schedules.py` · `train_grpo_schedule_w_val.py` · `grpo_schedule` (at root) ·
`sd3_sampler_ours.py` (training sampler) · `iqa_reward.py` · `run_train.sh` ·
`init_load_file_fin/`, `Datasets/`, `exp/inp_masks/`.
