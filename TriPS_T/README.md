# TriPS-T — Template schedule search

Training-free search for the triadic schedule. `solve.py` runs **one** configuration
(`function_dc / function_cfg / function_sto` ∈ `{linear, logarithm, exponential}` +
`step_scale → step_scale_2`), evaluates it, and appends a row (incl. `eval_score`) to
`summary_metrics.xlsx`. `run_search.sh` loops over all template combinations — the best
row is the confirmed TriPS-T schedule.

> `run_search.sh` adds the repo root to `PYTHONPATH` and `cd`s here, so the shared modules
> (`util`, `cores`, `functions`, `sd3_sampler_total`, `eval`, `custom_util` at the root) import fine.

## Dataset
Place the evaluation images under `./datasets/` in two subfolders:

```
datasets/
├── FFHQ_1000/     # 00000.png ... 00999.png   (1000 FFHQ images)
└── DIV2K_800/     # 0001.png  ... 0800.png    (800 DIV2K images)
```

- **FFHQ** — download from the official [FFHQ repository](https://github.com/NVlabs/ffhq-dataset),
  take `00000.png`–`00999.png` (the first 1000 images) and put them in `datasets/FFHQ_1000/`.
- **DIV2K** — download `DIV2K_train_HR` from the official
  [DIV2K page](https://data.vision.ee.ethz.ch/cvl/DIV2K/), take `0001.png`–`0800.png`
  (the first 800 images) and put them in `datasets/DIV2K_800/`.

## Grid search (→ Excel)

```bash
bash run_search.sh 0   # SR x8        (FFHQ)
bash run_search.sh 1   # gaussian deblur (FFHQ)
bash run_search.sh 2   # motion deblur   (DIV2K)
```

Pick the row with the best `eval_score` in `summary_metrics.xlsx`.

## Single (manual) run

```bash
python solve.py \
    --img_path <images> --prompt "a high quality photo of a face" \
    --method TriPS_T --task sr_bicubic --operator_imp SVD --deg_scale 8 \
    --img_size 768 --NFE 28 --cfg_scale 2.0 --seed 42 --inner_steps 6 \
    --step_scale 250 --step_scale_2 40 --stochasticity_weight 1.0 \
    --function_dc linear --function_cfg logarithm --function_sto logarithm \
    --workdir workdir_TriPS_T/srx8 --efficient_memory
```

## From a found schedule → inference / GRPO

- **Inference** with the found schedule: use the root [`inference.py`](../inference.py) (`--method TriPS-T`).
- **Export** the confirmed schedule to a GRPO reference policy: [`../TriPS_G/build_init_schedules.py`](../TriPS_G/build_init_schedules.py).

## Files

`solve.py` (single-run + Excel logging) · `run_search.sh` (template grid).
