#!/usr/bin/env bash
# ============================================================================
# TriPS-G  ·  demo runner (uses the shared ../demo_images folder)
# ----------------------------------------------------------------------------
# TriPS-G refines the TriPS-T template schedules with GRPO. This demo shows the
# full loop on a handful of demo images, initialised from the TriPS-T reference
# policy stored in init_load_file_fin/*.npz (regenerate them anytime with
# `python build_init_schedules.py --force`).
#
#   # 1) Short GRPO TRAINING demo (default mode), task in {0,1,2}:
#   bash run_demo_TriPS_G_demo.sh 0            # super-resolution x8 (bicubic)
#   bash run_demo_TriPS_G_demo.sh 1            # gaussian deblur
#   bash run_demo_TriPS_G_demo.sh 2            # motion deblur
#
#   # 2) INFERENCE with a trained schedule checkpoint:
#   GRPO_CKPT=/path/to/grpo_schedule_ckpt_xxx.pt \
#       bash run_demo_TriPS_G_demo.sh 0 test
#
# Notes
#   * Training here uses tiny iters/group_size/batch and offline W&B so it runs
#     quickly as a smoke test; use run_demo_TriPS_G_train.sh for the real config.
#   * Inference requires a checkpoint (.pt) produced by training; point GRPO_CKPT
#     at it (or edit the path below).
# ============================================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"
cd "$SCRIPT_DIR"

TASK_ID="${1:-0}"
MODE="${2:-train}"

DIV2K_DEMO="../demo_images/DIV2K"
DIV2K_PROMPTS="../demo_images/DIV2K/DIV2K_prompts_demo.txt"

# ---- task table: id -> (task, deg_scale, operator flag, init npz) -----------
case "${TASK_ID}" in
    0) TASK="sr_bicubic";    DEG=8;  OPERATOR="--operator_imp SVD"
       INIT="init_load_file_fin/DIV2K_sr_bicubic_sigma_eta_cfg_dc_ref_policy_fin.npz" ;;
    1) TASK="deblur_gauss";  DEG=3;  OPERATOR="--operator_imp SVD"
       INIT="init_load_file_fin/DIV2K_Gauss_deblur_sigma_eta_cfg_dc_ref_policy_fin.npz" ;;
    2) TASK="deblur_motion"; DEG=61; OPERATOR=""
       INIT="init_load_file_fin/DIV2K_motion_deblur_sigma_eta_cfg_dc_ref_policy_fin.npz" ;;
    *) echo "Usage: bash run_demo_TriPS_G_demo.sh [0|1|2] [train|test]"; exit 1 ;;
esac

# Make sure the TriPS-T reference policy (init npz) exists.
if [ ! -f "${INIT}" ]; then
    echo "[info] ${INIT} not found -> generating from TriPS-T template schedules ..."
    python build_init_schedules.py --dataset DIV2K --task "${TASK}" --force
fi

if [ "${MODE}" == "test" ]; then
    ########## INFERENCE with a trained GRPO schedule ########################
    GRPO_CKPT="${GRPO_CKPT:-/path/to/grpo_schedule_ckpt.pt}"
    if [ ! -f "${GRPO_CKPT}" ]; then
        echo "[error] set GRPO_CKPT to a trained checkpoint, e.g.:"
        echo "        GRPO_CKPT=workdir_TriPS_G_demo_${TASK}/ckpts/grpo_schedule_ckpt_${TASK}_it0001.pt \\"
        echo "            bash run_demo_TriPS_G_demo.sh ${TASK_ID} test"
        exit 1
    fi
    python solve_ours.py \
        --img_size 768 \
        --img_path ${DIV2K_DEMO} \
        --workdir workdir_TriPS_G_demo_${TASK}/infer \
        --prompt_file ${DIV2K_PROMPTS} \
        --method flowdps_moon \
        --task ${TASK} \
        ${OPERATOR} \
        --deg_scale ${DEG} \
        --noise_std 0.03 \
        --cfg_scale 2.0 \
        --seed 42 \
        --inner_steps 6 \
        --grpo_ckpt ${GRPO_CKPT} \
        --NFE 28 \
        --efficient_memory;
else
    ########## SHORT GRPO TRAINING demo ######################################
    python train_grpo_schedule_w_val.py \
        --img_size 768 \
        --img_path ${DIV2K_DEMO} \
        --workdir workdir_TriPS_G_demo_${TASK} \
        --init_load_file ${INIT} \
        --prompt_file ${DIV2K_PROMPTS} \
        --method TriPS_G_train \
        --task ${TASK} \
        ${OPERATOR} \
        --deg_scale ${DEG} \
        --noise_std 0.03 \
        --cfg_scale 2.0 \
        --seed 42 \
        --group_size 2 \
        --img_batch 2 \
        --reward_runs 1 \
        --iters 4 \
        --clip_eps 0.2 \
        --kl_beta 1e-3 \
        --no_kl_beta_adapt \
        --lr 1e-2 \
        --update_epochs 2 \
        --reward_mode modern_iqa \
        --reward_iqa_resize 768 \
        --reward_nr_view_mode resize \
        --reward_distortion_weight 0.3 --reward_perception_weight 0.7 \
        --reward_use_psnr --reward_use_lpips --reward_use_clip_iqa \
        --reward_lpips_patch 768 --reward_lpips_stride 0 \
        --reward_w_lpips 1.0 \
        --kappa 50 \
        --degree 25 \
        --NFE 28 \
        --efficient_memory \
        --ckpt_every 1 \
        --wandb_mode offline;
fi
