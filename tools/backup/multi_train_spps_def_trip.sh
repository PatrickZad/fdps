#! /bin/bash
port=$1
nc=$2
for data in prw cuhk # cdps
do
    for head in msroifc msroicoseg
    do
        count=0
        while(( ${count}<6 ))
        do
            let "count++"
            echo "Launch attemp "${count}" !"
            cd /home/patrick/Workspace/transearcher
            python tools/train_spps_baseline.py --config-file configs/search/spps/def/${data}/spps-${data}_${head}-oim.yaml \
            --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 10 \
            REID_HEAD.LOSS.METRIC.NAME triplet \
            REID_HEAD.LOSS.METRIC.NORM_FEAT True \
            OUTPUT_DIR outputs_expd/refine/spps/def/spps-${data}_${head}-oim-triplet
        done
    done
done