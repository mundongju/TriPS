#!/usr/bin/env bash
# ============================================================================
# TriPS · patch-FID / patch-KID evaluation (FLAIR-style non-overlap tiling)
#
#   bash run_eval_patch.sh fid <label_dir> <recon_dir> [patch_size] [max_images]
#   bash run_eval_patch.sh kid <label_dir> <recon_dir> [patch_size]
#
# Example (after a demo run):
#   bash run_eval_patch.sh fid results/TriPS-T/sr_bicubic/label results/TriPS-T/sr_bicubic/recon 256
# ============================================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

MODE="${1:-fid}"
LABEL_DIR="${2:?label_dir required}"
RECON_DIR="${3:?recon_dir required}"
PATCH_SIZE="${4:-256}"
MAX_IMAGES="${5:-1000}"

if [ "${MODE}" == "fid" ]; then
    python compute_patch_FID.py \
        --label_dir "${LABEL_DIR}" --recon_dir "${RECON_DIR}" \
        --patch_size "${PATCH_SIZE}" --max_images "${MAX_IMAGES}"
elif [ "${MODE}" == "kid" ]; then
    python compute_patch_KID.py \
        --label_dir "${LABEL_DIR}" --recon_dir "${RECON_DIR}" \
        --patch_size "${PATCH_SIZE}" --kid_subset_size 1000 --kid_subsets 50
else
    echo "Usage: bash run_eval_patch.sh [fid|kid] <label_dir> <recon_dir> [patch_size] [max_images]"
    exit 1
fi
