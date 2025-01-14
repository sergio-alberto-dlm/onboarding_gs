#!/bin/bash
set -euo pipefail
set -x

GPU_ID=0
DATA_BASE_PATH="/media/jbhayet/Data/datasets/BOP_CHALLENGE/hope/onboarding_static"
DATASETS=(
    hope
)

OBJECT_IDS=(
    4
    5
)

N_VIEWS=(
    20
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
                    --use_masks \
                    --face ${FACE}"

                    echo "========= ${DATASET}: Point cloud estimation (${FACE}) ========="
                    $CMD_PC
                else
                    echo "Keyframe output directory does not exist: ${KF_OUTPUT_PATH}"
                fi
            done

            DATA_BASE_PATH_PCD="./data/${DATASET}/obj_$(printf '%06d' ${OBJECT_ID})"
            # ----- Point Cloud Alignment -----
            CMD_PC_A="python ./pcd_align.py \
            --source_path=${DATA_BASE_PATH_PCD}/down/${N_VIEW}_views/sparse/0/points3D.ply \
            --target_path=${DATA_BASE_PATH_PCD}/up/${N_VIEW}_views/sparse/0/points3D.ply \
            --output_path=${DATA_BASE_PATH_PCD}/align \
            --masked_target_path=${DATA_BASE_PATH_PCD}/up/${N_VIEW}_views/masked_images \
            --masked_source_path=${DATA_BASE_PATH_PCD}/down/${N_VIEW}_views/masked_images \
            --target_camera_path=${DATA_BASE_PATH_PCD}/up/${N_VIEW}_views/sparse/0/cameras.txt \
            --source_camera_path=${DATA_BASE_PATH_PCD}/down/${N_VIEW}_views/sparse/0/cameras.txt \
            --target_image_path=${DATA_BASE_PATH_PCD}/up/${N_VIEW}_views/sparse/0/images.txt \
            --source_image_path=${DATA_BASE_PATH_PCD}/down/${N_VIEW}_views/sparse/0/images.txt"

            echo "========= ${DATASET}: point cloud aligning ========="
            $CMD_PC_A
        done

        # # ----- GS training --------
        # CMD_GS="CUDA_VISIBLE_DEVICES=${GPU_ID} python simple_trainer.py default \
        # --data_dir /home/sergio/onboarding_stage/data/${DATASET}/obj_$(printf '%06d' ${OBJECT_ID})/align \
        # --data_factor 1   \
        # --result_dir ./results/${DATASET}/obj_$(printf '%06d' ${OBJECT_ID})"
    done
done
