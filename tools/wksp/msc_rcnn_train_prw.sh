count=0
while(( ${count}<8 ))
do
    let "count++"
    echo "Launch attemp "${count}" !"
    python tools/train_seqnet.py --config-file configs/search/msc_rcnn/bseline/prw_mscreid_baseline.yaml --num-gpus 1 --resume --dist-url tcp://127.0.0.1:56688 --n-find-unparams OUTPUT_DIR outputs/msc_rcnn/prw_res5_sep2 SOLVER.IMS_PER_BATCH 5
done