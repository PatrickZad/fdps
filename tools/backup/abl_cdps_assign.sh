#! /bin/bash
port=$1
nc=$2
score=$3
iou=$4
# 0-05, 02-07, 10-05
for data in cdps #prw
do
        count=0
        while(( ${count}<6 ))
        do
            let "count++"
            echo "Launch attemp "${count}" !"
            cd /home/patrick/Workspace/transearcher
            python tools/train_spps_baseline_apk.py --config-file configs/search/spps/def/${data}/spps-${data}_msroicoseg-oim_b8_16e.yaml \
            --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 8 \
            REID_HEAD.LOSS.METRIC.NAME triplet \
            REID_HEAD.LOSS.METRIC.NORM_FEAT True \
            REID_HEAD.ID_ASSIGN.POS_SCORE_THRED ${score} \
            REID_HEAD.ID_ASSIGN.POS_IOU_THRED ${iou} \
            TEST.EVAL_PERIOD 57855 \
            APK_DROP_LAST True \
            OUTPUT_DIR outputs/refine/spps/def/spps-${data}_msroicoseg-oim-triplet_apk_conf${score}iou${iou}
        done
done