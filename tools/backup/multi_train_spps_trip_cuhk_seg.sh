port=$1
nc=$2

count=0
while(( ${count}<8 ))
do
    let "count++"
    echo "Launch attemp "${count}" !"
    cd /home/patrick/Workspace/transearcher
    python tools/train_spps_baseline_apk.py --config-file configs/search/spps/spps-cuhk_msroicoseg-oim-triplet.yaml \
    --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 10 \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    OUTPUT_DIR outputs/refine/spps/spps-cuhk_msroicoseg-oim-triplet-norm_apk
done

count=9
while(( ${count}<8 ))
do
    let "count++"
    echo "Launch attemp "${count}" !"
    cd /home/patrick/Workspace/transearcher
    python tools/train_spps_baseline.py --config-file configs/search/spps/spps-cuhk_msroicoseg-oim-triplet.yaml \
    --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 10 \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    OUTPUT_DIR outputs/refine/spps/spps-cuhk_msroicoseg-oim-triplet-norm
done
