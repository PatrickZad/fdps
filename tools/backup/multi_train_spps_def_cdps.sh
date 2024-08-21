#! /bin/bash
data=cdps
port=$1
nc=$2
#base 1500, 0.2, 0.7
for head in msroicoseg # roifc roicoseg
do
    count=0
    while(( ${count}<6 ))
    do
        let "count++"
        echo "Launch attemp "${count}" !"
        cd /home/patrick/Workspace/transearcher
        python tools/train_spps_baseline.py --config-file configs/search/spps/def/${data}/spps-${data}_${head}-oim_b8_16e.yaml \
        --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 8 \
        OUTPUT_DIR outputs/refine/spps/def/spps-${data}_${head}-oim_b8_16e
    done
done
for head in msroicoseg # roifc roicoseg
do
    count=0
    while(( ${count}<8 ))
    do
        let "count++"
        echo "Launch attemp "${count}" !"
        cd /home/patrick/Workspace/transearcher
        python tools/train_spps_baseline.py --config-file configs/search/spps/def/${data}/spps-${data}_${head}-oim_b8_16e.yaml \
        --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 8 \
        OUTPUT_DIR outputs_expd/refine/spps/def/spps-${data}_${head}-oim_b8_16e_re
    done
done
for head in msroifc # roifc roicoseg
do
    count=0
    while(( ${count}<8 ))
    do
        let "count++"
        echo "Launch attemp "${count}" !"
        cd /home/patrick/Workspace/transearcher
        python tools/train_spps_baseline.py --config-file configs/search/spps/def/${data}/spps-${data}_${head}-oim_b8_16e.yaml \
        --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} --n-find-unparams SOLVER.IMS_PER_BATCH 8 \
        OUTPUT_DIR outputs_expd/refine/spps/def/spps-${data}_${head}-oim_b8_16e_re
    done
done

# single scale
"""
for head in roifc msroifc roicoseg msroicoseg
do
    count=0
    while(( ${count}<8 ))
    do
        let "count++"
        echo "Launch attemp "${count}" !"
        cd /home/patrick/Workspace/transearcher
        python tools/train_spps_baseline.py --config-file configs/search/spps/def/${data}/spps-${data}_${head}-oim.yaml \
        --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 10 INPUT.MIN_SIZE_TRAIN 900,900 \
        OUTPUT_DIR outputs/refine/spps/def/spps-${data}_${head}-oim_sscale
    done
done
"""