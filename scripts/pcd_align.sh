#! /bin/bash

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

# Number of training iterations (if needed in the future steps)
gs_train_iter=15000

for DATASET in "${DATASETS[@]}"; do
    for OBJECT_ID in "${OBJECT_IDS[@]}"; do
        for N_VIEW in "${N_VIEWS[@]}"; do

            SOURCE_PATH = "data/${DATASET}/obj_${OBJECT_ID:06d}/down/${N_VIEW}_views                # ----- (1) Key-frame selection -----
                CMD_PCD_A="python ./pcd_align.py \
                --data_base_path ${DATA_BASE_PATH} \
                --dataset_name ${DATASET} \
                --object_id ${OBJECT_ID} \
                --n_views ${N_VIEW}"

                echo "========= ${DATASET}: Key-frame selection (${FACE}) ========="
                eval $CMD_KF
        done
    done
done

python pcd_align.py --source_path=./data/hope/obj_000001/down/7_views/sparse/0/points3D.ply --target_path=./data/hope/obj_000001/up/7_views/sparse/0/points3D.ply --output_path=./data/hope/obj_000001/align --masked_target_path=./data/hope/obj_000001/up/7_views/masked_images --masked_source_path=./data/hope/obj_000001/down/7_views/masked_images --target_camera_path=./data/hope/obj_000001/up/7_views/sparse/0/cameras.txt --source_camera_path=./data/hope/obj_000001/down/7_views/sparse/0/cameras.txt --target_image_path=./data/hope/obj_000001/up/7_views/sparse/0/images.txt --source_image_path=./data/hope/obj_000001/down/7_views/sparse/0/images.txt
