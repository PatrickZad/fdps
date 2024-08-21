port=$1
nc=$2
#! /bin/bash
for cfg in 3lr
do
        count=0
        while(( ${count}<6 ))
        do
            let "count++"
            echo "Launch attemp "${count}" !"
            cd /home/patrick/Workspace/transearcher
            python tools/train_spps_baseline.py --config-file configs/search/spps/spps-prw_msroicoseg-oim_${cfg}.yaml \
            --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 10 \
            OUTPUT_DIR outputs/refine/spps/spps-prw_msroicoseg-oim_${cfg}
        done
done