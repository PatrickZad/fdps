port=$1
count=0
while(( ${count}<6 ))
do
    let "count++"
    echo "Launch attemp "${count}" !"
    cd /home/patrick/Workspace/transearcher
    python tools/train_dtps_baseline.py --config-file configs/search/dtps/prw/dtps-prw_dtdec-oim_d5r1_clip.yaml \
    --num-gpus 2 --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 8
done
