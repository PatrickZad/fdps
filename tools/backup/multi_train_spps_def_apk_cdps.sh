#! /bin/bash
port=$1
nc=$2
for data in cdps # prw cuhk
do
    for head in msroifc # roicoseg  roifc msroicoseg
    do
        count=0
        while(( ${count}<0 ))
        do
            let "count++"
            echo "Launch attemp "${count}" !"
            cd /home/patrick/Workspace/transearcher
            python tools/train_spps_baseline_apk.py --config-file configs/search/spps/def/${data}/spps-${data}_${head}-oim_b8_16e.yaml \
            --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} \
            REID_HEAD.LOSS.METRIC.NAME triplet \
            REID_HEAD.LOSS.METRIC.NORM_FEAT True \
            APK_DROP_LAST True \
            OUTPUT_DIR outputs_expd/refine/spps/def/spps-${data}_${head}-oim-triplet_apk_b8_16e_re
        done
    done
done
for data in cdps # prw cuhk
do
    for head in msroicoseg # msroifc roicoseg  roifc
    do
        count=0
        while(( ${count}<6 ))
        do
            let "count++"
            echo "Launch attemp "${count}" !"
            cd /home/patrick/Workspace/transearcher
            python tools/train_spps_baseline_apk.py --config-file configs/search/spps/def/${data}/spps-${data}_${head}-oim_b8_16e.yaml \
            --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 8 \
            REID_HEAD.LOSS.METRIC.NAME triplet \
            REID_HEAD.LOSS.METRIC.NORM_FEAT True \
            APK_DROP_LAST True \
            OUTPUT_DIR outputs_expd/refine/spps/def/spps-${data}_${head}-oim-triplet_apk_b8_16e_re
        done
    done
done
#b4
