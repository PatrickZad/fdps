port=$1
nc=$2
#! /bin/bash
for cfg in p3 p4
do
        count=0
        while(( ${count}<6 ))
        do
            let "count++"
            echo "Launch attemp "${count}" !"
            cd /home/patrick/Workspace/transearcher
            python tools/train_srcnn_param-g.py --config-file configs/search/srcnn_fine/srcnn_trid_cuhk_w30_lulb_2lr_${cfg}.yaml \
            --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port}
        done
done