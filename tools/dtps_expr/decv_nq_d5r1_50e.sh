port=$1
count=0
while(( ${count}<6 ))
do
    let "count++"
    echo "Launch attemp "${count}" !"
    cd /home/patrick/Workspace/transearcher
    python tools/train_dtps_baseline.py --config-file configs/search/dtps/prw/dtps-prw_dtdecv-oim_d5r1_nq_25e.yaml \
    --num-gpus 2 --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 8 \
    EXT_VIS.W_AND_B.PROJECT decv_nq_d5r1_50e \
    OUTPUT_DIR outputs/dtps/decv_nq_d5r1_50e
done