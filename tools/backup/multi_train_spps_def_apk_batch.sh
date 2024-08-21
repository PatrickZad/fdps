#! /bin/bash
port=$1
nc=$2
data=$3
for head in msroicoseg # msroifc roicoseg  roifc
do
    count=0
    while(( ${count}<0 ))
    do
            let "count++"
            echo "Launch attemp "${count}" !"
            cd /home/patrick/Workspace/transearcher
            python tools/train_spps_baseline_apk.py --config-file configs/search/spps/def/${data}/spps-${data}_${head}-oim_b5.yaml \
            --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} \
            REID_HEAD.LOSS.METRIC.NAME triplet \
            REID_HEAD.LOSS.METRIC.NORM_FEAT True \
            SOLVER.BASE_LR 0.000025 \
            OUTPUT_DIR outputs/refine/spps/def/spps-${data}_${head}-oim-triplet_apk_batch
    done
done
port=$1
nc=$2
data=$3
for head in msroicoseg # msroifc roicoseg  roifc
do
    count=0
    while(( ${count}<3 ))
    do
            let "count++"
            echo "Launch attemp "${count}" !"
            cd /home/patrick/Workspace/transearcher
            python tools/train_spps_baseline_apk.py --config-file configs/search/spps/def/${data}/spps-${data}_${head}-oim_b5.yaml \
            --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} \
            REID_HEAD.LOSS.METRIC.NAME triplet \
            REID_HEAD.LOSS.METRIC.NORM_FEAT True \
            SOLVER.BASE_LR 0.00005 \
            OUTPUT_DIR outputs/refine/spps/def/spps-${data}_${head}-oim-triplet_apk_batch_lr
    done
done
#APK_DROP_LAST True \