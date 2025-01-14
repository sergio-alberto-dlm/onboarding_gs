#! /bin/bash

GPU_ID=0
DATASETS=(
    hope
)

OBJECT_IDS=(
    5
)

for DATASET in "${DATASETS[@]}"; do
    for OBJECT_ID in "${OBJECT_IDS[@]}"; do

            # ----- Gaussian Splatting Training -----
            CMD_GS_T="CUDA_VISIBLE_DEVICES=0 python gaussian_splatting/simple_trainer.py default \
            --data_dir data/${DATASET}/obj_$(printf '%06d' ${OBJECT_ID})/align         
            --data_factor 1           
            --result_dir gaussian_splatting/results/${DATASET}/obj_$(printf '%06d' ${OBJECT_ID})
            --max_steps 15_000 
            --eval_steps 5_000 15_000 
            --save_steps 15_000 
            --scale_reg 10.0 
            --pose-opt" 
            
            echo "========= ${DATASET} obj_$(printf '%06d' ${OBJECT_ID}): gaussian splatting training ========="
            eval $CMD_GS_T

    done
done

