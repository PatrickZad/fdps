cfg=$1
ds=$2
port=$3
nc=$4
#! /bin/bash
for oim_w in 10 30 16
do
    for ulb in 'ulb' 'ulb_full'
    do
        count=0
        while(( ${count}<6 ))
        do
            let "count++"
            echo "Launch attemp "${count}" !"
            cd /home/patrick/Workspace/transearcher
            python tools/train_srcnn.py --config-file ${cfg} \
            --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} \
            MODEL.SEARCH.PERSON_FEAT.OIM.LABELED_WEIGHT ${oim_w} \
            MODEL.SEARCH.PERSON_FEAT.OIM.UNLABELED_WEIGHT ${oim_w} \
            MODEL.SEARCH.PERSON_FEAT.OIM.ULB_LAYER ${ulb} \
            OUTPUT_DIR outputs/refine/srcnn_trid_w_ulb_${ds}_w${oim_w}_l${ulb}
        done
    done
done