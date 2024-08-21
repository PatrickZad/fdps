port=$1
nc=$2
#! /bin/bash
for cfg in roi32-8 roi14-14 
do
        count=0
        while(( ${count}<6 ))
        do
            let "count++"
            echo "Launch attemp "${count}" !"
            cd /home/patrick/Workspace/transearcher
            python tools/train_srcnn.py --config-file configs/search/srcnn_fine/srcnn_trid_w_ulb_prw_w30_lulb_2lr_${cfg}.yaml \
            --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port}
        done
done