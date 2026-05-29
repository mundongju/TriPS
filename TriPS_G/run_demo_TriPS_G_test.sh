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

# # ####### FFHQ ###########
FFHQ_DIR_1="/home/dongju1126/dataset_ICML/FFHQ_10"
FFHQ_DIR_2="/home/dongju1126/dataset_ICML/FFHQ_Analysis_25"
FFHQ_DIR_fin="/home/dongju1126/dataset_ICML/FFHQ_1000"
# FFHQ_DIR="/home/qkdwnstj10/ICML26/Datasets/FFHQ_200_GRPO"




PROMPT_FILE_1="DIV2K_prompts_set10.txt"
PROMPT_FILE_2="DIV2K_prompts_set25.txt"
PROMPT_FILE_3="DIV2K_prompts_valid_10.txt"
PROMPT_FILE_fin="DIV2K_prompts.txt"
DIV2K_DIR_1="/home/dongju1126/dataset_ICML/DIV2K_10"
DIV2K_DIR_2="/home/dongju1126/dataset_ICML/DIV2K_25"
DIV2K_DIR_3="/home/dongju1126/dataset_ICML/DIV2K_GRPO_train_10"
DIV2K_DIR_fin="/home/dongju1126/dataset_ICML/DIV2K_train_HR"


########## super resolution x8 task DIV2K #################
if [ $1 == 0 ]; then

    grpo_ckpt_path="/home/dongju1126/Ours_GRPO_final_v4/QAlign_real_FINAL_sr_bicubic_x8_DIV2K_random_batch_reward_ablation_NO_RESIZE_iqa_dw_0.3_pw_0.7_psnr_lpips_clip_iqa_QALIGN_DIV2K_GRPO_train_100_lr_1e-2_gs4_bs4_rr3_w_val/ckpts/grpo_schedule_ckpt_sr_bicubic_it0160.pt"
    python solve_ours.py \
        --img_size 768 \
        --img_path ${DIV2K_DIR_fin} \
        --workdir workdir_Ours_results/sr_bicubic/QAlign_real_FINAL_sr_bicubic_x8_DIV2K_random_batch_reward_ablation_NO_RESIZE_iqa_dw_0.3_pw_0.7_psnr_lpips_clip_iqa_QALIGN_DIV2K_GRPO_train_100_lr_1e-2_gs4_bs4_rr3_w_val_160pt/DIV2K_train_HR \
        --prompt_file ${PROMPT_FILE_fin} \
        --method flowdps_moon \
        --task sr_bicubic \
        --operator_imp SVD \
        --deg_scale 8 \
        --noise_std 0.03 \
        --cfg_scale 2.0 \
        --seed 42 \
        --inner_steps 6 \
        --grpo_ckpt ${grpo_ckpt_path} \
        --NFE 28 \
        --efficient_memory;

fi

########## motion deblur task DIV2K #################
if [ $1 == 1 ]; then

    grpo_ckpt_path="/home/dongju1126/Ours_GRPO_final_v4/QAlign_real_FINAL_deblur_motion_DIV2K_random_batch_reward_ablation_NO_RESIZE_iqa_dw_0.3_pw_0.7_psnr_lpips_clip_iqa_QALIGN_DIV2K_GRPO_train_100_lr_1e-2_gs4_bs4_rr3_w_val/ckpts/grpo_schedule_ckpt_deblur_motion_it0117.pt"
    python solve_ours.py \
        --img_size 768 \
        --img_path ${DIV2K_DIR_fin} \
        --workdir workdir_Ours_results/sr_bicubic/QAlign_real_FINAL_deblur_motion_DIV2K_random_batch_reward_ablation_NO_RESIZE_iqa_dw_0.3_pw_0.7_psnr_lpips_clip_iqa_QALIGN_DIV2K_GRPO_train_100_lr_1e-2_gs4_bs4_rr3_w_val_117pt/DIV2K_train_HR \
        --prompt_file ${PROMPT_FILE_fin} \
        --method flowdps_moon \
        --task deblur_motion \
        --deg_scale 61 \
        --noise_std 0.03 \
        --cfg_scale 2.0 \
        --seed 42 \
        --inner_steps 6 \
        --grpo_ckpt ${grpo_ckpt_path} \
        --NFE 28 \
        --efficient_memory;

fi

########## gaussian deblur task DIV2K #################
if [ $1 == 2 ]; then

    grpo_ckpt_path="/home/dongju1126/Ours_GRPO_final_v4/QAlign_real_FINAL_deblur_gauss_sigma_3_DIV2K_random_batch_reward_ablation_dw_0.3_pw_0.7_psnr_lpips_clip_iqa_QALIGN_DIV2K_GRPO_train_100_lr_1e-2_gs4_bs4_rr3_w_val/ckpts/grpo_schedule_ckpt_deblur_gauss_it0170.pt"
    python solve_ours.py \
        --img_size 768 \
        --img_path ${DIV2K_DIR_fin} \
        --workdir workdir_Ours_results/sr_bicubic/QAlign_real_FINAL_deblur_gauss_sigma_3_DIV2K_random_batch_reward_ablation_dw_0.3_pw_0.7_psnr_lpips_clip_iqa_QALIGN_DIV2K_GRPO_train_100_lr_1e-2_gs4_bs4_rr3_w_val_170pt/DIV2K_train_HR \
        --prompt_file ${PROMPT_FILE_fin} \
        --method flowdps_moon \
        --task deblur_gauss \
        --operator_imp SVD \
        --deg_scale 3 \
        --noise_std 0.03 \
        --cfg_scale 2.0 \
        --seed 42 \
        --inner_steps 6 \
        --grpo_ckpt ${grpo_ckpt_path} \
        --NFE 28 \
        --efficient_memory;

fi