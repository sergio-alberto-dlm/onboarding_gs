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

gs_train_iter=15000

for DATASET in "${DATASETS[@]}"; do
    for OBJECT_ID in "${OBJECT_IDS[@]}"; do
        for N_VIEW in "${N_VIEWS[@]}"; do

            # ----- (1) key-frame selection -----
            CMD_KF="python ./kf_selection.py \
            --data_base_path ${DATA_BASE_PATH} \
            --dataset_name ${DATASET} \
            --object_id ${OBJECT_ID} \
            --n_views ${N_VIEW}"

            echo "========= ${DATASET}: key-frame selection ========="
            eval $CMD_KF
        done
    done
done
