#!/bin/bash

GPU_ID=0
DATA_BASE_PATH="/media/jbhayet/Data/datasets/BOP_CHALLENGE/hope/onboarding_static"
DATASETS=(
    hope
)

OBJECT_IDS=(
    1
)

N_VIEWS=(
    7
)

FACES=(
    down
    up
)

# Number of training iterations
gs_train_iter=15000

for DATASET in "${DATASETS[@]}"; do
    for OBJECT_ID in "${OBJECT_IDS[@]}"; do
        for N_VIEW in "${N_VIEWS[@]}"; do
            for FACE in "${FACES[@]}"; do
                # ----- (1) Key-frame selection -----
                CMD_KF="python ./kf_selection.py \
                --data_base_path ${DATA_BASE_PATH} \
                --dataset_name ${DATASET} \
                --object_id ${OBJECT_ID} \
                --n_views ${N_VIEW} \
                --face ${FACE}"

                echo "========= ${DATASET}: Key-frame selection (${FACE}) ========="
                $CMD_KF

                # ----- (2) Point cloud estimation -----
                KF_OUTPUT_PATH="./data/${DATASET}/obj_$(printf '%06d' ${OBJECT_ID})/${FACE}/${N_VIEW}_views"
                
                if [ -d "${KF_OUTPUT_PATH}" ]; then
                    CMD_PC="python ./coarse_geometry.py \
                    --focal_avg \
                    --n_views ${N_VIEW} \
                    --img_base_path ${KF_OUTPUT_PATH} \
                    --use_masks"

                    echo "========= ${DATASET}: Point cloud estimation (${FACE}) ========="
                    $CMD_PC
                else
                    echo "Keyframe output directory does not exist: ${KF_OUTPUT_PATH}"
                fi
            done
        done
    done
done
