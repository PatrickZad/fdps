l1_w=$1
giou_w=$2
port=$3
count=0
while(( ${count}<6 ))
do
    let "count++"
    echo "Launch attemp "${count}" !"
    cd /home/patrick/Workspace/transearcher
    python tools/train_spps_baseline.py --config-file configs/search/spps/def/prw/spps-prw_roifc-oim.yaml \
    --num-gpus 2 --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 10 \
    OUTPUT_DIR outputs/det_expr/spps-prw_msroifc-oim-giou${giou_w}-l1${l1_w}
done