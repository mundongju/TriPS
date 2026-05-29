# TriPS-T — Training-free Template Solver

Part of **[TriPS](../README.md)** (ICML 2026). TriPS-T solves inverse problems with
**no training** by scheduling the triad (DC / CFG / stochasticity) through
*functional-prior templates* (`linear` / `logarithm` / `exponential`).

> Run scripts here automatically add the repo **root** to `PYTHONPATH` (so the shared
> `util.py`, `cores/`, `functions/` are importable) and `cd` into this folder. Just run them.

## Quick demo

```bash
bash run_demo_TriPS_T_demo.sh 0   # super-resolution x8 (bicubic), FFHQ
bash run_demo_TriPS_T_demo.sh 1   # gaussian deblur,               FFHQ
bash run_demo_TriPS_T_demo.sh 2   # motion deblur,                 DIV2K
```

## Manual run

```bash
python solve.py \
    --img_size 768 --img_path ../demo_images/FFHQ \
    --prompt "a high quality photo of a face" \
    --method TriPS_T --task sr_bicubic --operator_imp SVD --deg_scale 8 \
    --noise_std 0.03 --cfg_scale 2.0 --seed 42 --NFE 28 \
    --step_scale 250 --step_scale_2 40 --inner_steps 6 --stochasticity_weight 1.0 \
    --function_dc linear --function_cfg logarithm --function_sto logarithm \
    --workdir workdir_TriPS_T_demo/SRx8 --efficient_memory
```

### Key arguments

| Argument | Meaning |
|---|---|
| `--task` | `sr_bicubic`, `deblur_gauss`, `deblur_motion`, `inpainting`, … |
| `--operator_imp` | forward-operator backend: `SVD` or `FFT` |
| `--deg_scale` | SR: downscale factor · deblur: kernel size |
| `--step_scale`, `--step_scale_2` | DC (data-consistency) start / end values |
| `--function_dc / _cfg / _sto` | template shape per triad component |
| `--stochasticity_weight` | scales the `sto='logarithm'` curve |
| `--NFE` | number of sampling steps (default 28) |
| `--efficient_memory` | offload text encoder (single 24 GB GPU) |

## Final confirmed schedules

The schedules selected by the TriPS-T template search (per dataset/task) are exported
to `../TriPS_G/init_load_file_fin/*.npz` via
[`../TriPS_G/build_init_schedules.py`](../TriPS_G/build_init_schedules.py); the same
values populate the demo runner above.

## Files

- `solve.py` — entry point
- `sd3_sampler_total.py` — SD3.5 sampler with the triadic template schedules
- `custom_util.py`, `eval.py` — helpers / metrics
- `compute_patch_FID_final.py`, `compute_patch_KID_final.py`, `pFID.sh` — evaluation
- `run_demo_TriPS_T.sh` — full reproduction grid · `run_demo_TriPS_T_demo.sh` — demo
- `inp_masks/`, `DIV2K_prompts.txt` — task assets
