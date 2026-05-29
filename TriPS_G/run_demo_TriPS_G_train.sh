#!/usr/bin/env bash
# ============================================================================
# TriPS path bootstrap (added during repo restructure for the public release).
# The common Python modules (util.py, cores/, functions/, motionblur/) now live
# at the repository ROOT, one level above this script. We expose the repo root
# on PYTHONPATH and run from this script's own directory so that every
# task-relative path (datasets/, Datasets/, exp/, inp_masks/, init_load_file_fin/,
# *.txt) keeps resolving exactly as before. No python command below is changed.
# ============================================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"
cd "$SCRIPT_DIR"
# ============================================================================

########## gaussian deblur task DIV2K #################
if [ $1 == 0 ]; then

    FFHQ_DIR="./Datasets/DIV2K_GRPO_train_100"
    FFHQ_DIR_VAL="./Datasets/DIV2K_GRPO_val_10"
    python train_grpo_schedule_w_val.py \
        --img_size 768 \
        --img_path ${FFHQ_DIR} \
        --val_img_path ${FFHQ_DIR_VAL} \
        --val_every 5 \
        --val_max_images 50 \
        --workdir QAlign_real_FINAL_deblur_gauss_sigma_3_DIV2K_random_batch_reward_ablation_dw_0.3_pw_0.7_psnr_lpips_clip_iqa_QALIGN_DIV2K_GRPO_train_100_lr_1e-2_gs4_bs4_rr3_w_val \
        --init_load_file init_load_file_fin/DIV2K_Gauss_deblur_sigma_eta_cfg_dc_ref_policy_fin.npz \
        --prompt_file ./DIV2K_prompts_valid_train_100.txt \
        --prompt_file_val ./DIV2K_prompts_valid_10.txt \
        --method TriPS_G_train \
        --task deblur_gauss \
        --operator_imp SVD \
        --deg_scale 3 \
        --noise_std 0.03 \
        --cfg_scale 2.0 \
        --seed 42 \
        --group_size 4 \
        --img_batch 4 \
        --reward_runs 3 \
        --iters 400 \
        --clip_eps 0.2 \
        --kl_beta 1e-3 \
        --no_kl_beta_adapt \
        --lr 1e-2 \
        --update_epochs 4 \
        --reward_mode modern_iqa \
        --reward_iqa_resize 224 \
        --reward_nr_view_mode five \
        --reward_nr_crop_native 384 \
        --reward_distortion_weight 0.3 --reward_perception_weight 0.7 \
        --reward_use_psnr --reward_use_lpips --reward_use_clip_iqa --reward_use_qalign \
        --reward_lpips_patch 768 --reward_lpips_stride 0 \
        --reward_w_lpips 1.0 \
        --reward_use_msssim \
        --reward_msssim_min 0.70 --reward_msssim_max 1.00 \
        --reward_w_msssim 0.0 \
        --kappa 50 \
        --degree 25 \
        --NFE 28 \
        --efficient_memory \
        --ckpt_every 1 \
        --use_wandb \
        --wandb_project grpo-schedule-beta-bernstein-demo-v1 \
        --wandb_mode online \
        # --resume
        # --no_reward_use_qalign

fi

########## super resolution x8 task DIV2K #################

if [ $1 == 1 ]; then

    FFHQ_DIR="./Datasets/DIV2K_GRPO_train_100"
    FFHQ_DIR_VAL="./Datasets/DIV2K_GRPO_val_10"
    python train_grpo_schedule_w_val.py \
        --img_size 768 \
        --img_path ${FFHQ_DIR} \
        --val_img_path ${FFHQ_DIR_VAL} \
        --val_every 5 \
        --val_max_images 50 \
        --workdir QAlign_real_FINAL_sr_bicubic_x8_DIV2K_random_batch_reward_ablation_NO_RESIZE_iqa_dw_0.3_pw_0.7_psnr_lpips_clip_iqa_QALIGN_DIV2K_GRPO_train_100_lr_1e-2_gs4_bs4_rr3_w_val \
        --init_load_file init_load_file_fin/DIV2K_sr_bicubic_sigma_eta_cfg_dc_ref_policy_fin.npz \
        --prompt_file ./DIV2K_prompts_valid_train_100.txt \
        --prompt_file_val ./DIV2K_prompts_valid_10.txt \
        --method TriPS_G_train \
        --task sr_bicubic \
        --operator_imp SVD \
        --deg_scale 8 \
        --noise_std 0.03 \
        --cfg_scale 2.0 \
        --seed 42 \
        --group_size 4 \
        --img_batch 4 \
        --reward_runs 3 \
        --iters 400 \
        --clip_eps 0.2 \
        --kl_beta 1e-3 \
        --no_kl_beta_adapt \
        --lr 1e-2 \
        --update_epochs 4 \
        --reward_mode modern_iqa \
        --reward_iqa_resize 768 \
        --reward_nr_view_mode resize \
        --reward_nr_crop_native 384 \
        --reward_distortion_weight 0.3 --reward_perception_weight 0.7 \
        --reward_use_psnr --reward_use_lpips --reward_use_clip_iqa --reward_use_qalign \
        --reward_lpips_patch 768 --reward_lpips_stride 0 \
        --reward_w_lpips 1.0 \
        --reward_use_msssim \
        --reward_msssim_min 0.70 --reward_msssim_max 1.00 \
        --reward_w_msssim 0.0 \
        --kappa 50 \
        --degree 25 \
        --NFE 28 \
        --efficient_memory \
        --ckpt_every 1 \
        --use_wandb \
        --wandb_project grpo-schedule-beta-bernstein-demo-v1 \
        --wandb_mode online \
        # --resume
        # --no_reward_use_qalign

fi

########## motion deblur task DIV2K #################
if [ $1 == 2 ]; then

    FFHQ_DIR="./Datasets/DIV2K_GRPO_train_100"
    FFHQ_DIR_VAL="./Datasets/DIV2K_GRPO_val_10"
    python train_grpo_schedule_w_val.py \
        --img_size 768 \
        --img_path ${FFHQ_DIR} \
        --val_img_path ${FFHQ_DIR_VAL} \
        --val_every 5 \
        --val_max_images 50 \
        --workdir QAlign_real_FINAL_deblur_motion_DIV2K_random_batch_reward_ablation_NO_RESIZE_iqa_dw_0.3_pw_0.7_psnr_lpips_clip_iqa_QALIGN_DIV2K_GRPO_train_100_lr_1e-2_gs4_bs4_rr3_w_val \
        --init_load_file init_load_file_fin/DIV2K_motion_deblur_sigma_eta_cfg_dc_ref_policy_fin.npz \
        --prompt_file ./DIV2K_prompts_valid_train_100.txt \
        --prompt_file_val ./DIV2K_prompts_valid_10.txt \
        --method TriPS_G_train \
        --task deblur_motion \
        --deg_scale 61 \
        --noise_std 0.03 \
        --cfg_scale 2.0 \
        --seed 42 \
        --group_size 4 \
        --img_batch 4 \
        --reward_runs 3 \
        --iters 400 \
        --clip_eps 0.2 \
        --kl_beta 1e-3 \
        --no_kl_beta_adapt \
        --lr 1e-2 \
        --update_epochs 4 \
        --reward_mode modern_iqa \
        --reward_iqa_resize 768 \
        --reward_nr_view_mode resize \
        --reward_nr_crop_native 384 \
        --reward_distortion_weight 0.3 --reward_perception_weight 0.7 \
        --reward_use_psnr --reward_use_lpips --reward_use_clip_iqa --reward_use_qalign \
        --reward_lpips_patch 768 --reward_lpips_stride 0 \
        --reward_w_lpips 1.0 \
        --reward_use_msssim \
        --reward_msssim_min 0.70 --reward_msssim_max 1.00 \
        --reward_w_msssim 0.0 \
        --kappa 50 \
        --degree 25 \
        --NFE 28 \
        --efficient_memory \
        --ckpt_every 1 \
        --use_wandb \
        --wandb_project grpo-schedule-beta-bernstein-demo-v1 \
        --wandb_mode online \
        --resume
        # --no_reward_use_qalign

fi