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
########## super resolution x8 task FFHQ #################
if [ $1 == 0 ]; then
    ARRAY_FUNCTION=('linear' 'logarithm' 'exponential')
    ARRAY_FUNCTION_2=('linear' 'logarithm' 'exponential')
    ARRAY_FUNCTION_3=('linear' 'logarithm' 'exponential')

    FFHQ_DIR="./datasets/FFHQ_1000"

    for function_dc in "${ARRAY_FUNCTION[@]}";
    do
        for function_cfg in "${ARRAY_FUNCTION_2[@]}";
        do
            for function_sto in "${ARRAY_FUNCTION_3[@]}";
            do
                python solve.py \
                    --img_size 768 \
                    --img_path ${FFHQ_DIR} \
                    --workdir workdir_TriPS_T_demo/SRx8/dc_${function_dc}_cfg_${function_cfg}_sto_${function_sto} \
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
                    --function_dc ${function_dc} \
                    --function_cfg ${function_cfg} \
                    --function_sto ${function_sto} \
                    --efficient_memory;
            done
        done
    done

########## gaussian deblur task FFHQ #################
elif [ $1 == 1 ]; then
    ARRAY_FUNCTION=('linear' 'logarithm' 'exponential')
    ARRAY_FUNCTION_2=('linear' 'logarithm' 'exponential')
    ARRAY_FUNCTION_3=('linear' 'logarithm' 'exponential')

    FFHQ_DIR="./datasets/FFHQ_1000"

    for function_dc in "${ARRAY_FUNCTION[@]}";
    do
        for function_cfg in "${ARRAY_FUNCTION_2[@]}";
        do
            for function_sto in "${ARRAY_FUNCTION_3[@]}";
            do
                python solve.py \
                    --img_size 768 \
                    --img_path ${FFHQ_DIR} \
                    --workdir workdir_TriPS_T_demo/gaussian_deblurring/dc_${function_dc}_cfg_${function_cfg}_sto_${function_sto} \
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
                    --function_dc ${function_dc} \
                    --function_cfg ${function_cfg} \
                    --function_sto ${function_sto} \
                    --efficient_memory;
            done
        done
    done

########## motion deblur task DIV2K #################
elif [ $1 == 2 ]; then
    ARRAY_FUNCTION=('linear')
    ARRAY_FUNCTION_2=('exponential')
    ARRAY_FUNCTION_3=('linear')

    DIV2K_DIR="./datasets/DIV2K_800"
    PROMPT_FILE="DIV2K_prompts.txt"

    for function_dc in "${ARRAY_FUNCTION[@]}";
    do
        for function_cfg in "${ARRAY_FUNCTION_2[@]}";
        do
            for function_sto in "${ARRAY_FUNCTION_3[@]}";
            do
                python solve.py \
                    --img_size 768 \
                    --img_path ${DIV2K_DIR} \
                    --workdir workdir_TriPS_T_demo/motion_deblurring/dc_${function_dc}_cfg_${function_cfg}_sto_${function_sto} \
                    --prompt_file ${PROMPT_FILE} \
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
                    --function_dc ${function_dc} \
                    --function_cfg ${function_cfg} \
                    --function_sto ${function_sto} \
                    --efficient_memory;
            done
        done
    done

fi