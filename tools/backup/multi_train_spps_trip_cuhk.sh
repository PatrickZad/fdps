port=$1
nc=$2
count=0
while(( ${count}<10 ))
do
    let "count++"
    echo "Launch attemp "${count}" !"
    cd /home/patrick/Workspace/transearcher
    python tools/train_spps_baseline.py --config-file configs/search/spps/spps-cuhk_msroifc-oim-triplet.yaml \
    --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 10
done
count=0
while(( ${count}<10 ))
do
    let "count++"
    echo "Launch attemp "${count}" !"
    cd /home/patrick/Workspace/transearcher
    python tools/train_spps_baseline_apk.py --config-file configs/search/spps/spps-cuhk_msroifc-oim-triplet.yaml \
    --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 10 \
    OUTPUT_DIR outputs/refine/spps/spps-cuhk_msroifc-oim-triplet_apk
done