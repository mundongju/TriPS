#!/usr/bin/env bash
# ============================================================================
# TriPS-T  ·  demo runner (uses the shared ../demo_images folder)
# ----------------------------------------------------------------------------
# Runs the *training-free* template solver (solve.py) on the bundled demo images
# with the FINAL CONFIRMED TriPS-T schedules (the same triadic schedules that are
# exported to TriPS_G/init_load_file_fin/*.npz by build_init_schedules.py).
#
#   bash run_demo_TriPS_T_demo.sh 0   # super-resolution x8 (bicubic), FFHQ faces
#   bash run_demo_TriPS_T_demo.sh 1   # gaussian deblur,            FFHQ faces
#   bash run_demo_TriPS_T_demo.sh 2   # motion deblur,              DIV2K scenes
#
# Outputs are written under ./workdir_TriPS_T_demo/<task>/ .
# ============================================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"
cd "$SCRIPT_DIR"

FFHQ_DEMO="../demo_images/FFHQ"
DIV2K_DEMO="../demo_images/DIV2K"
DIV2K_PROMPTS="../demo_images/DIV2K/DIV2K_prompts_demo.txt"

########## (0) super-resolution x8 (bicubic) — FFHQ #########################
if [ "$1" == "0" ]; then
    python solve.py \
        --img_size 768 \
        --img_path ${FFHQ_DEMO} \
        --workdir workdir_TriPS_T_demo/SRx8 \
        --prompt "a high quality photo of a face" \
        --method TriPS_T \
        --task sr_bicubic \
        --operator_imp SVD \
        --deg_scale 8 \
        --noise_std 0.03 \
        --cfg_scale 2.0 \
        --seed 42 \
        --step_scale 250 \
        --step_scale_2 50 \
        --inner_steps 6 \
        --stochasticity_weight 1.0 \
        --NFE 28 \
        --function_dc linear \
        --function_cfg logarithm \
        --function_sto logarithm \
        --efficient_memory;

########## (1) gaussian deblur — FFHQ #######################################
elif [ "$1" == "1" ]; then
    python solve.py \
        --img_size 768 \
        --img_path ${FFHQ_DEMO} \
        --workdir workdir_TriPS_T_demo/gaussian_deblur \
        --prompt "a high quality photo of a face" \
        --method TriPS_T \
        --task deblur_gauss \
        --operator_imp SVD \
        --deg_scale 3 \
        --noise_std 0.03 \
        --cfg_scale 2.0 \
        --seed 42 \
        --step_scale 200 \
        --step_scale_2 100 \
        --inner_steps 6 \
        --stochasticity_weight 1.0 \
        --NFE 28 \
        --function_dc logarithm \
        --function_cfg logarithm \
        --function_sto logarithm \
        --efficient_memory;

########## (2) motion deblur — DIV2K ########################################
elif [ "$1" == "2" ]; then
    python solve.py \
        --img_size 768 \
        --img_path ${DIV2K_DEMO} \
        --workdir workdir_TriPS_T_demo/motion_deblur \
        --prompt_file ${DIV2K_PROMPTS} \
        --method TriPS_T \
        --task deblur_motion \
        --operator_imp FFT \
        --deg_scale 61 \
        --noise_std 0.03 \
        --cfg_scale 2.0 \
        --seed 42 \
        --step_scale 350 \
        --step_scale_2 150 \
        --inner_steps 6 \
        --stochasticity_weight 1.0 \
        --NFE 28 \
        --function_dc linear \
        --function_cfg exponential \
        --function_sto linear \
        --efficient_memory;

else
    echo "Usage: bash run_demo_TriPS_T_demo.sh [0|1|2]"
    echo "  0 = super-resolution x8 (bicubic), FFHQ"
    echo "  1 = gaussian deblur,               FFHQ"
    echo "  2 = motion deblur,                 DIV2K"
fi
