#!/usr/bin/env bash
# ============================================================================
# TriPS · unified demo inference on ./demo_images  (run from the repo root)
#
#   bash run_inference.sh TriPS-T 0     # TriPS-T, super-resolution x8 (FFHQ)
#   bash run_inference.sh TriPS-T 1     # TriPS-T, gaussian deblur     (FFHQ)
#   bash run_inference.sh TriPS-T 2     # TriPS-T, motion deblur       (DIV2K)
#
#   GRPO_CKPT=/path/to/ckpt.pt bash run_inference.sh TriPS-G 0   # TriPS-G (DIV2K)
#
# Results -> ./results/<method>/<task>/{recon,label,input1,...}
# ============================================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH}"

METHOD="${1:-TriPS-T}"     # TriPS-T | TriPS-G
TASK_ID="${2:-0}"          # 0 = sr_bicubic | 1 = deblur_gauss | 2 = deblur_motion

case "${TASK_ID}" in
    0) TASK="sr_bicubic" ;;
    1) TASK="deblur_gauss" ;;
    2) TASK="deblur_motion" ;;
    *) echo "task id must be 0|1|2"; exit 1 ;;
esac

# demo data: faces (FFHQ) for sr/gauss, scenes (DIV2K) for motion
if [ "${TASK}" == "deblur_motion" ]; then
    DATASET="DIV2K"; IMG="demo_images/DIV2K"; PROMPT_ARG="--prompt_file demo_images/DIV2K/DIV2K_prompts_demo.txt"
else
    DATASET="FFHQ";  IMG="demo_images/FFHQ";  PROMPT_ARG="--prompt \"a high quality photo of a face\""
fi

if [ "${METHOD}" == "TriPS-T" ]; then
    eval python inference.py \
        --method TriPS-T --dataset ${DATASET} --task ${TASK} \
        --img_path ${IMG} ${PROMPT_ARG} \
        --workdir results/TriPS-T/${TASK} \
        --seed 42 --NFE 28 --inner_steps 6 --efficient_memory

elif [ "${METHOD}" == "TriPS-G" ]; then
    GRPO_CKPT="${GRPO_CKPT:-}"
    if [ -z "${GRPO_CKPT}" ] || [ ! -f "${GRPO_CKPT}" ]; then
        echo "[error] TriPS-G needs a trained checkpoint. Train one with TriPS_G/run_train.sh,"
        echo "        then:  GRPO_CKPT=TriPS_G/<workdir>/ckpts/grpo_schedule_ckpt_${TASK}_itXXXX.pt \\"
        echo "                   bash run_inference.sh TriPS-G ${TASK_ID}"
        exit 1
    fi
    # TriPS-G demo always uses DIV2K scenes
    eval python inference.py \
        --method TriPS-G --task ${TASK} \
        --img_path demo_images/DIV2K --prompt_file demo_images/DIV2K/DIV2K_prompts_demo.txt \
        --grpo_ckpt "${GRPO_CKPT}" \
        --workdir results/TriPS-G/${TASK} \
        --seed 42 --NFE 28 --inner_steps 6 --efficient_memory
else
    echo "Usage: bash run_inference.sh [TriPS-T|TriPS-G] [0|1|2]"
    exit 1
fi
