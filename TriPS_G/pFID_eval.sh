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
# diffusion 
# python compute_patch_FID_final.py \
#   --label_dir /home/dongju1126/code_diffusion/TReg/workdir_DDPG_MD_final/Final/label \
#   --recon_dir  /home/dongju1126/code_diffusion/TReg/workdir_DDPG_MD_final/Final/recon \
#   --max_images 1000 \
#   --patch_size 128 \
#   --patches_per_image 16 \
#   --patch_seed 42 \
#   --batch_size 32 \
#   --device cuda

# python compute_patch_FID_final.py \
#   --label_dir /home/dongju1126/code_diffusion/TReg/workdir_DDPG_SR_final/Final/label \
#   --recon_dir  /home/dongju1126/code_diffusion/TReg/workdir_DDPG_SR_final/Final/recon \
#   --max_images 1000 \
#   --patch_size 128 \
#   --patches_per_image 16 \
#   --patch_seed 42 \
#   --batch_size 32 \
#   --device cuda

# flow
# python compute_patch_FID_final.py \
#   --label_dir /home/qkdwnstj10/ICML26/Ours_GRPO_final_v3/workdir_Ours_results/gauss_deblur/workdir_gauss_deblur_DIV2K_random_batch_reward_ablation_dw_0.3_pw_0.7_psnr_lpips_no_resizes_89pt/DIV2K_train_HR/label \
#   --recon_dir /home/qkdwnstj10/ICML26/Ours_GRPO_final_v3/workdir_Ours_results/gauss_deblur/workdir_gauss_deblur_DIV2K_random_batch_reward_ablation_dw_0.3_pw_0.7_psnr_lpips_no_resizes_89pt/DIV2K_train_HR/recon \
#   --max_images 800 \
#   --patch_size 256 \
#   --patches_per_image 9 \
#   --patch_seed 42 \
#   --batch_size 32 \
#   --device cuda

# python compute_patch_FID_final.py \
#   --label_dir /home/qkdwnstj10/ICML26/Ours_GRPO_final_v3/workdir_Ours_results/gauss_deblur/workdir_gauss_deblur_FFHQ_random_batch_reward_ablation_dw_0.3_pw_0.7_psnr_lpips_no_resizes_83pt/FFHQ_1000/label \
#   --recon_dir /home/qkdwnstj10/ICML26/Ours_GRPO_final_v3/workdir_Ours_results/gauss_deblur/workdir_gauss_deblur_FFHQ_random_batch_reward_ablation_dw_0.3_pw_0.7_psnr_lpips_no_resizes_83pt/FFHQ_1000/recon \
#   --max_images 1000 \
#   --patch_size 256 \
#   --patches_per_image 9 \
#   --patch_seed 42 \
#   --batch_size 32 \
#   --device cuda

# python compute_patch_FID_final.py \
#   --label_dir /home/qkdwnstj10/ICML26/Ours_GRPO_final_v3/workdir_Ours_results/gauss_deblur/workdir_gauss_deblur_FFHQ_random_batch_reward_ablation_dw_0.3_dp_0.7_ssim_0_psnr_1_lpips_1_clipiqa_1_qalign_1_55pt/FFHQ_1000/label \
#   --recon_dir /home/qkdwnstj10/ICML26/Ours_GRPO_final_v3/workdir_Ours_results/gauss_deblur/workdir_gauss_deblur_FFHQ_random_batch_reward_ablation_dw_0.3_dp_0.7_ssim_0_psnr_1_lpips_1_clipiqa_1_qalign_1_55pt/FFHQ_1000/recon \
#   --max_images 1000 \
#   --patch_size 256 \
#   --patches_per_image 9 \
#   --patch_seed 42 \
#   --batch_size 32 \
#   --device cuda

# python compute_patch_FID_final.py \
#   --label_dir /home/dongju1126/Ours_GRPO_final_v4/workdir_Ours_results/sr_bicubic/workdir_sr_bicubic_DIV2K_random_batch_reward_ablation_dw_0.2_pw_0.8_psnr_lpips_no_resizes_clip_iqa_QALIGN_DIV2K_GRPO_train_20_lr_1e-2_80pt/DIV2K_train_HR/label \
#   --recon_dir /home/dongju1126/Ours_GRPO_final_v4/workdir_Ours_results/sr_bicubic/workdir_sr_bicubic_DIV2K_random_batch_reward_ablation_dw_0.2_pw_0.8_psnr_lpips_no_resizes_clip_iqa_QALIGN_DIV2K_GRPO_train_20_lr_1e-2_80pt/DIV2K_train_HR/recon \
#   --max_images 800 \
#   --patch_size 256 \
#   --patches_per_image 9 \
#   --patch_seed 42 \
#   --batch_size 32 \
#   --device cuda

# python compute_patch_FID_final.py \
#   --label_dir /home/dongju1126/Ours_GRPO_final_v4/workdir_Ours_results/sr_bicubic/workdir_sr_bicubic_DIV2K_random_batch_reward_ablation_dw_0.2_pw_0.8_psnr_lpips_no_resizes_clip_iqa_QALIGN_DIV2K_GRPO_train_10_lr_1e-2_53pt/DIV2K_train_HR/label \
#   --recon_dir /home/dongju1126/Ours_GRPO_final_v4/workdir_Ours_results/sr_bicubic/workdir_sr_bicubic_DIV2K_random_batch_reward_ablation_dw_0.2_pw_0.8_psnr_lpips_no_resizes_clip_iqa_QALIGN_DIV2K_GRPO_train_10_lr_1e-2_53pt/DIV2K_train_HR/recon \
#   --max_images 800 \
#   --patch_size 256 \
#   --patches_per_image 9 \
#   --patch_seed 42 \
#   --batch_size 32 \
#   --device cuda

# python compute_patch_FID_final.py \
#   --label_dir /home/dongju1126/Ours_GRPO_final_v4/workdir_Ours_results/sr_bicubic/workdir_sr_bicubic_DIV2K_random_batch_reward_ablation_dw_0.0_pw_1.0_psnr_lpips_no_resizes_clip_iqa_QALIGN_DIV2K_GRPO_train_10_lr_1e-2_92pt/DIV2K_train_HR/label \
#   --recon_dir /home/dongju1126/Ours_GRPO_final_v4/workdir_Ours_results/sr_bicubic/workdir_sr_bicubic_DIV2K_random_batch_reward_ablation_dw_0.0_pw_1.0_psnr_lpips_no_resizes_clip_iqa_QALIGN_DIV2K_GRPO_train_10_lr_1e-2_92pt/DIV2K_train_HR/recon \
#   --max_images 800 \
#   --patch_size 256 \
#   --patches_per_image 9 \
#   --patch_seed 42 \
#   --batch_size 32 \
#   --device cuda

# python compute_patch_FID_final.py \
#   --label_dir /home/dongju1126/Ours_GRPO_final_v4/workdir_Ours_results/sr_bicubic/QAlign_real_FINAL_deblur_gauss_sigma_3_DIV2K_random_batch_reward_ablation_dw_0.3_pw_0.7_psnr_lpips_clip_iqa_QALIGN_DIV2K_GRPO_train_100_lr_1e-2_gs4_bs4_rr3_w_val_170pt/DIV2K_train_HR/label \
#   --recon_dir /home/dongju1126/Ours_GRPO_final_v4/workdir_Ours_results/sr_bicubic/QAlign_real_FINAL_deblur_gauss_sigma_3_DIV2K_random_batch_reward_ablation_dw_0.3_pw_0.7_psnr_lpips_clip_iqa_QALIGN_DIV2K_GRPO_train_100_lr_1e-2_gs4_bs4_rr3_w_val_170pt/DIV2K_train_HR/recon \
#   --max_images 800 \
#   --patch_size 256 \
#   --patches_per_image 9 \
#   --patch_seed 42 \
#   --batch_size 32 \
#   --device cuda

# python compute_patch_FID_final.py \
#   --label_dir /home/dongju1126/Ours_GRPO_final_v4/workdir_Ours_results/sr_bicubic/QAlign_real_FINAL_sr_bicubic_x8_DIV2K_random_batch_reward_ablation_dw_0.3_pw_0.7_psnr_lpips_clip_iqa_QALIGN_DIV2K_GRPO_train_100_lr_1e-2_gs4_bs4_rr3_w_val_200pt/DIV2K_train_HR/label \
#   --recon_dir /home/dongju1126/Ours_GRPO_final_v4/workdir_Ours_results/sr_bicubic/QAlign_real_FINAL_sr_bicubic_x8_DIV2K_random_batch_reward_ablation_dw_0.3_pw_0.7_psnr_lpips_clip_iqa_QALIGN_DIV2K_GRPO_train_100_lr_1e-2_gs4_bs4_rr3_w_val_200pt/DIV2K_train_HR/recon \
#   --max_images 800 \
#   --patch_size 256 \
#   --patches_per_image 9 \
#   --patch_seed 42 \
#   --batch_size 32 \
#   --device cuda


# python compute_patch_FID_final.py \
#   --label_dir /home/dongju1126/Ours_GRPO_final_v4/workdir_Ours_results/sr_bicubic/workdir_sr_bicubic_DIV2K_random_batch_reward_ablation_dw_0.8_pw_0.2_psnr_lpips_no_resizes_2.0_clip_iqa_QALIGN_DIV2K_GRPO_train_10_lr_1e-2_gs4_bs4_rr3_w_val_75pt/DIV2K_train_HR/label \
#   --recon_dir /home/dongju1126/Ours_GRPO_final_v4/workdir_Ours_results/sr_bicubic/workdir_sr_bicubic_DIV2K_random_batch_reward_ablation_dw_0.8_pw_0.2_psnr_lpips_no_resizes_2.0_clip_iqa_QALIGN_DIV2K_GRPO_train_10_lr_1e-2_gs4_bs4_rr3_w_val_75pt/DIV2K_train_HR/recon \
#   --max_images 800 \
#   --patch_size 256 \
#   --patches_per_image 9 \
#   --patch_seed 42 \
#   --batch_size 32 \
#   --device cuda

# python compute_patch_FID_final.py \
#   --label_dir /home/dongju1126/Ours_GRPO_final_v4/workdir_Ours_results/sr_bicubic/workdir_sr_bicubic_DIV2K_random_batch_reward_ablation_dw_0.4_pw_0.6_psnr_lpips_no_resizes_clip_iqa_QALIGN_DIV2K_GRPO_train_4_lr_1e-2_gs_2_bat_4_run_3_final/DIV2K_train_HR/label \
#   --recon_dir /home/dongju1126/Ours_GRPO_final_v4/workdir_Ours_results/sr_bicubic/workdir_sr_bicubic_DIV2K_random_batch_reward_ablation_dw_0.4_pw_0.6_psnr_lpips_no_resizes_clip_iqa_QALIGN_DIV2K_GRPO_train_4_lr_1e-2_gs_2_bat_4_run_3_final/DIV2K_train_HR/recon \
#   --max_images 800 \
#   --patch_size 256 \
#   --patches_per_image 9 \
#   --patch_seed 42 \
#   --batch_size 32 \
#   --device cuda

# #### Ours + GRPO ##########################
# #### SR DIV2K ##########################
############20260119 no resize IQA#################
# python compute_patch_FID_final.py \
#   --label_dir /home/dongju1126/Ours_GRPO_final_v4/workdir_Ours_results/sr_bicubic/QAlign_real_FINAL_deblur_motion_DIV2K_random_batch_reward_ablation_dw_0.3_pw_0.7_psnr_lpips_clip_iqa_QALIGN_DIV2K_GRPO_train_100_lr_1e-2_gs4_bs4_rr3_w_val_219pt/DIV2K_train_HR/label \
#   --recon_dir /home/dongju1126/Ours_GRPO_final_v4/workdir_Ours_results/sr_bicubic/QAlign_real_FINAL_deblur_motion_DIV2K_random_batch_reward_ablation_dw_0.3_pw_0.7_psnr_lpips_clip_iqa_QALIGN_DIV2K_GRPO_train_100_lr_1e-2_gs4_bs4_rr3_w_val_219pt/DIV2K_train_HR/recon \
#   --max_images 800 \
#   --patch_size 256 \
#   --patches_per_image 9 \
#   --patch_seed 42 \
#   --batch_size 32 \
#   --device cuda

# #### Ours + GRPO ##########################
# #### MD DIV2K ##########################
python compute_patch_FID_final.py \
  --label_dir /home/dongju1126/Ours_GRPO_final_v4/workdir_Ours_results/sr_bicubic/QAlign_real_FINAL_deblur_motion_DIV2K_random_batch_reward_ablation_NO_RESIZE_iqa_dw_0.3_pw_0.7_psnr_lpips_clip_iqa_QALIGN_DIV2K_GRPO_train_100_lr_1e-2_gs4_bs4_rr3_w_val_117pt/DIV2K_train_HR/label \
  --recon_dir /home/dongju1126/Ours_GRPO_final_v4/workdir_Ours_results/sr_bicubic/QAlign_real_FINAL_deblur_motion_DIV2K_random_batch_reward_ablation_NO_RESIZE_iqa_dw_0.3_pw_0.7_psnr_lpips_clip_iqa_QALIGN_DIV2K_GRPO_train_100_lr_1e-2_gs4_bs4_rr3_w_val_117pt/DIV2K_train_HR/recon \
  --max_images 800 \
  --patch_size 256 \
  --patches_per_image 9 \
  --patch_seed 42 \
  --batch_size 32 \
  --device cuda

python compute_patch_FID_final.py \
  --label_dir /home/dongju1126/Ours_GRPO_final_v4/workdir_Ours_results/sr_bicubic/QAlign_real_FINAL_deblur_motion_DIV2K_random_batch_reward_ablation_NO_RESIZE_iqa_dw_0.3_pw_0.7_psnr_lpips_clip_iqa_QALIGN_DIV2K_GRPO_train_100_lr_1e-2_gs4_bs4_rr3_w_val_117pt/DIV2K_train_HR/label \
  --recon_dir /home/dongju1126/Ours_GRPO_final_v4/workdir_Ours_results/sr_bicubic/QAlign_real_FINAL_sr_bicubic_x8_DIV2K_random_batch_reward_ablation_NO_RESIZE_iqa_dw_0.3_pw_0.7_psnr_lpips_clip_iqa_QALIGN_DIV2K_GRPO_train_100_lr_1e-2_gs4_bs4_rr3_w_val_60pt/DIV2K_train_HR/recon \
  --max_images 800 \
  --patch_size 256 \
  --patches_per_image 9 \
  --patch_seed 42 \
  --batch_size 32 \
  --device cuda

python compute_patch_FID_final.py \
  --label_dir /home/dongju1126/Ours_GRPO_final_v4/workdir_Ours_results/sr_bicubic/QAlign_real_FINAL_deblur_motion_DIV2K_random_batch_reward_ablation_NO_RESIZE_iqa_dw_0.3_pw_0.7_psnr_lpips_clip_iqa_QALIGN_DIV2K_GRPO_train_100_lr_1e-2_gs4_bs4_rr3_w_val_117pt/DIV2K_train_HR/label \
  --recon_dir /home/dongju1126/Ours_GRPO_final_v4/workdir_Ours_results/sr_bicubic/QAlign_real_FINAL_sr_bicubic_x8_DIV2K_random_batch_reward_ablation_NO_RESIZE_iqa_dw_0.3_pw_0.7_psnr_lpips_clip_iqa_QALIGN_DIV2K_GRPO_train_100_lr_1e-2_gs4_bs4_rr3_w_val_100pt/DIV2K_train_HR/recon \
  --max_images 800 \
  --patch_size 256 \
  --patches_per_image 9 \
  --patch_seed 42 \
  --batch_size 32 \
  --device cuda