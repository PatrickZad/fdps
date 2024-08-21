port=$1
nc=$2
#! /bin/bash
for cfg in 5 7 6
do
        count=0
        while(( ${count}<6 ))
        do
            let "count++"
            echo "Launch attemp "${count}" !"
            cd /home/patrick/Workspace/transearcher
            python tools/train_spps_baseline.py --config-file configs/search/spps/spps-prw_roicoseg-oim_new.yaml \
            --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 10 \
            REID_HEAD.PERSON_FEATURE.PART.NUM_PARTS ${cfg} \
            OUTPUT_DIR outputs/refine/spps/spps-prw_roicoseg-oim_new_${cfg}part
        done
done