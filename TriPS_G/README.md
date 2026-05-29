# TriPS-G — GRPO Schedule Optimization

Part of **[TriPS](../README.md)** (ICML 2026). TriPS-G refines the TriPS-T template
schedules into **flexible temporal curves** with **GRPO** (Group Relative Policy
Optimization), driven by a hybrid IQA reward (perceptual + distortion) and KL-anchored
to the TriPS-T reference policy.

> Run scripts here automatically add the repo **root** to `PYTHONPATH` (so the shared
> `util.py`, `cores/`, `functions/` are importable) and `cd` into this folder. Just run them.

## Pipeline

```
TriPS-T schedules ──(build_init_schedules.py)──▶ init_load_file_fin/*.npz
        │                                                  │
        │                                       (reference policy / init)
        ▼                                                  ▼
                       train_grpo_schedule_w_val.py  ──▶  ckpts/*.pt
                                                           │
                                                  (trained schedule)
                                                           ▼
                                                     solve_ours.py  ──▶  reconstructions
```

## 1) Build the reference policy (TriPS-T → npz)

The 6 reference-policy files (`{DIV2K,FFHQ} × {sr_bicubic, Gauss_deblur, motion_deblur}`)
ship with the repo and can be regenerated deterministically. Each holds four length-`NFE`
arrays: `sigma`, `eta` (stochasticity), `cfg`, `step_scale` (data-consistency).

```bash
python build_init_schedules.py --verify     # confirm shipped npz match the templates
python build_init_schedules.py --force      # regenerate all 6
python build_init_schedules.py --dataset DIV2K --task sr_bicubic --force   # just one
```

## 2) Train

```bash
# short smoke-test on ../demo_images (offline W&B, tiny iters):
bash run_demo_TriPS_G_demo.sh 0             # 0=sr_bicubic | 1=deblur_gauss | 2=deblur_motion

# full configuration:
bash run_demo_TriPS_G_train.sh 1            # 0=gauss | 1=sr | 2=motion
```

Key flags (see `train_grpo_schedule_w_val.py` for all): `--init_load_file` (reference
policy npz), `--group_size`, `--img_batch`, `--reward_runs`, `--iters`, `--lr`,
`--kl_beta`, `--clip_eps`, the `--reward_*` family, and `--use_wandb`/`--wandb_mode`.
Checkpoints are saved under `<workdir>/ckpts/`.

## 3) Inference

```bash
python solve_ours.py \
    --img_size 768 --img_path ../demo_images/DIV2K \
    --prompt_file ../demo_images/DIV2K/DIV2K_prompts_demo.txt \
    --method flowdps_moon --task sr_bicubic --operator_imp SVD --deg_scale 8 \
    --noise_std 0.03 --cfg_scale 2.0 --seed 42 --NFE 28 --inner_steps 6 \
    --grpo_ckpt /path/to/grpo_schedule_ckpt_sr_bicubic_itXXXX.pt \
    --workdir workdir_TriPS_G_test/sr_bicubic --efficient_memory
```

Or via the demo runner: `GRPO_CKPT=... bash run_demo_TriPS_G_demo.sh 0 test`.

## Evaluation

```bash
bash pFID_eval.sh        # patch-FID over the produced reconstructions
```

## Files

- `train_grpo_schedule_w_val.py` — GRPO training (with validation)
- `grpo_schedule.py` — schedule parameterization (`coeff_to_schedule`, `ScheduleBounds`, …)
- `iqa_reward.py` — `ModernIQAReward` (PSNR / LPIPS / CLIP-IQA / Q-Align / MS-SSIM)
- `solve_ours.py` — inference with a trained schedule
- `sd3_sampler_ours.py`, `sd3_sampler_ours_test.py`, `eval.py`
- `build_init_schedules.py` — export TriPS-T schedules → `init_load_file_fin/*.npz`
- `init_load_file_fin/` — 6 reference-policy `.npz`
- `Datasets/` — GRPO train/val image subsets · `exp/inp_masks/` — inpainting masks
- `run_demo_TriPS_G_train.sh`, `run_demo_TriPS_G_test.sh`, `run_demo_TriPS_G_demo.sh`

> **Note:** the original `run_demo_TriPS_G_test.sh` is kept as a server-side reference and
> contains absolute paths / a `--inner_steps_2` flag that `solve_ours.py` does not define;
> prefer `run_demo_TriPS_G_demo.sh` (or the manual command above) for a clean inference run.
