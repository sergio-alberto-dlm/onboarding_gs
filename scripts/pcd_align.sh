#! /bin/bash

GPU_ID=0
DATA_ROOT_PATH="/media/jbhayet/Data/datasets/BOP_CHALLENGE/hope/onboarding_static"
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


for DATASET in "${DATASETS[@]}"; do
    for OBJECT_ID in "${OBJECT_IDS[@]}"; do
        for N_VIEW in "${N_VIEWS[@]}"; do

            DATA_BASE_PATH="./data/${DATASET}/obj_$(printf '%06d' ${OBJECT_ID})"
            # ----- Point Cloud Alignment -----
            CMD_PCD_A="python ./pcd_align.py \
            --source_path=${DATA_BASE_PATH}/down/${N_VIEW}_views/sparse/0/points3D.ply \
            --target_path=${DATA_BASE_PATH}/up/${N_VIEW}_views/sparse/0/points3D.ply \
            --output_path=${DATA_BASE_PATH}/align \
            --masked_target_path=${DATA_BASE_PATH}/up/${N_VIEW}_views/masked_images \
            --masked_source_path=${DATA_BASE_PATH}/down/${N_VIEW}_views/masked_images \
            --target_camera_path=${DATA_BASE_PATH}/up/${N_VIEW}_views/sparse/0/cameras.txt \
            --source_camera_path=${DATA_BASE_PATH}/down/${N_VIEW}_views/sparse/0/cameras.txt \
            --target_image_path=${DATA_BASE_PATH}/up/${N_VIEW}_views/sparse/0/images.txt \
            --source_image_path=${DATA_BASE_PATH}/down/${N_VIEW}_views/sparse/0/images.txt"

            echo "========= ${DATASET}: Key-frame selection ========="
            eval $CMD_PCD_A
        done
    done
done

