port=$1
nc=$2

count=0
while(( ${count}<0 ))
do
    let "count++"
    echo "Launch attemp "${count}" !"
    cd /home/patrick/Workspace/transearcher
    python tools/train_spps_baseline_apk.py --config-file configs/search/spps/def/ref/cuhk_droplast.yaml \
    --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 10 \
    REID_HEAD.LOSS.METRIC.NAME triplet \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    OUTPUT_DIR outputs_expd/refine/spps/def/spps-cuhk_msroicoseg-oim-triplet_apk_droplast
done

count=0
while(( ${count}<0 ))
do
    let "count++"
    echo "Launch attemp "${count}" !"
    cd /home/patrick/Workspace/transearcher
    python tools/train_spps_baseline_apk.py --config-file configs/search/spps/def/ref/cuhk_dconv.yaml \
    --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 10 \
    REID_HEAD.LOSS.METRIC.NAME triplet \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    INPUT.SIZE_DIVISIBILITY 32 \
    APK_DROP_LAST True \
    OUTPUT_DIR outputs_expd/refine/spps/def/spps-cuhk_msroicoseg-oim-triplet_apk_droplast_dconv_32
done

count=0
while(( ${count}<8 ))
do
    let "count++"
    echo "Launch attemp "${count}" !"
    cd /home/patrick/Workspace/transearcher
    python tools/train_spps_baseline_apk.py --config-file configs/search/spps/def/ref/cuhk_500.yaml \
    --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 10 \
    REID_HEAD.LOSS.METRIC.NAME triplet \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    INPUT.SIZE_DIVISIBILITY 32 \
    APK_DROP_LAST True \
    OUTPUT_DIR outputs_expd/refine/spps/def/spps-cuhk_msroicoseg-oim-triplet_apk_droplast_500
done

count=0
while(( ${count}<0 ))
do
    let "count++"
    echo "Launch attemp "${count}" !"
    cd /home/patrick/Workspace/transearcher
    python tools/train_spps_baseline.py --config-file configs/search/spps/def/ref/cuhk_gap.yaml \
    --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 10
done

count=0
while(( ${count}<0 ))
do
    let "count++"
    echo "Launch attemp "${count}" !"
    cd /home/patrick/Workspace/transearcher
    python tools/train_spps_baseline.py --config-file configs/search/spps/def/ref/cuhk_syncbn.yaml \
    --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 10
done

count=0
while(( ${count}<0 ))
do
    let "count++"
    echo "Launch attemp "${count}" !"
    cd /home/patrick/Workspace/transearcher
    python tools/train_spps_baseline.py --config-file configs/search/spps/def/ref/cuhk_nclip.yaml \
    --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 10
done



count=0
while(( ${count}<0 ))
do
    let "count++"
    echo "Launch attemp "${count}" !"
    cd /home/patrick/Workspace/transearcher
    python tools/train_spps_baseline.py --config-file configs/search/spps/def/ref/cuhk_1504.yaml \
    --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 10
done

