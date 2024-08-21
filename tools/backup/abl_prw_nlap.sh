#! /bin/bash
port=$1
nc=$2
for data in prw # cdps
do
        count=0
        while(( ${count}<0 ))
        do
            let "count++"
            echo "Launch attemp "${count}" !"
            cd /home/patrick/Workspace/transearcher
            python tools/train_spps_baseline_apk.py --config-file configs/search/spps/def/${data}/spps-${data}_msroicoseg-oim.yaml \
            --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 10 \
            REID_HEAD.LOSS.METRIC.NAME triplet \
            REID_HEAD.LOSS.METRIC.NORM_FEAT True \
            REID_HEAD.LOSS.LOSS_WEIGHTS.NON_LAP 0.1 \
            TEST.EVAL_PERIOD 41040 \
            OUTPUT_DIR outputs_expd/refine/spps/def/spps-${data}_msroicoseg-oim-triplet_apk_01nlap
        done
        count=0
        while(( ${count}<0 ))
        do
            let "count++"
            echo "Launch attemp "${count}" !"
            cd /home/patrick/Workspace/transearcher
            python tools/train_spps_baseline_apk.py --config-file configs/search/spps/def/${data}/spps-${data}_msroicoseg-oim.yaml \
            --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 10 \
            REID_HEAD.LOSS.METRIC.NAME triplet \
            REID_HEAD.LOSS.METRIC.NORM_FEAT True \
            REID_HEAD.LOSS.LOSS_WEIGHTS.NON_LAP 0.3 \
            TEST.EVAL_PERIOD 41040 \
            INPUT.SIZE_DIVISIBILITY 32 \
            OUTPUT_DIR outputs_expd/refine/spps/def/spps-${data}_msroicoseg-oim-triplet_apk_03nlap
        done
        count=0
        while(( ${count}<0 ))
        do
            let "count++"
            echo "Launch attemp "${count}" !"
            cd /home/patrick/Workspace/transearcher
            python tools/train_spps_baseline_apk.py --config-file configs/search/spps/def/${data}/spps-${data}_msroicoseg-oim.yaml \
            --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 10 \
            REID_HEAD.LOSS.METRIC.NAME triplet \
            REID_HEAD.LOSS.METRIC.NORM_FEAT True \
            REID_HEAD.LOSS.LOSS_WEIGHTS.NON_LAP 0.7 \
            TEST.EVAL_PERIOD 41040 \
            INPUT.SIZE_DIVISIBILITY 32 \
            OUTPUT_DIR outputs_expd/refine/spps/def/spps-${data}_msroicoseg-oim-triplet_apk_07nlap
        done
        count=0
        while(( ${count}<6 ))
        do
            let "count++"
            echo "Launch attemp "${count}" !"
            cd /home/patrick/Workspace/transearcher
            python tools/train_spps_baseline_apk.py --config-file configs/search/spps/def/${data}/spps-${data}_msroicoseg-oim.yaml \
            --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 10 \
            REID_HEAD.LOSS.METRIC.NAME triplet \
            REID_HEAD.LOSS.METRIC.NORM_FEAT True \
            REID_HEAD.LOSS.LOSS_WEIGHTS.NON_LAP 0.9 \
            TEST.EVAL_PERIOD 41040 \
            INPUT.SIZE_DIVISIBILITY 32 \
            OUTPUT_DIR outputs_expd/refine/spps/def/spps-${data}_msroicoseg-oim-triplet_apk_09nlap
        done
done