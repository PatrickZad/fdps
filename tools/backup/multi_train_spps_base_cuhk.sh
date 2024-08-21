port=$1
nc=$2
#! /bin/bash
for cfg in roifc msroifc
do
        count=0
        while(( ${count}<10 ))
        do
            let "count++"
            echo "Launch attemp "${count}" !"
            cd /home/patrick/Workspace/transearcher
            python tools/train_spps_baseline.py --config-file configs/search/spps/spps-cuhk_${cfg}-oim_newbase.yaml \
            --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 10
        done
done