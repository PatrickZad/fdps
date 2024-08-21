count=0
while(( ${count}<8 ))
do
    let "count++"
    echo "Launch attemp "${count}" !"
    python tools/train_seqnet.py --config-file configs/search/msc_rcnn/bseline/cuhk_mscreid_baseline.yaml --num-gpus 1 --resume --dist-url tcp://127.0.0.1:58899 --n-find-unparams OUTPUT_DIR outputs/msc_rcnn/cuhk_res5_sep2 SOLVER.IMS_PER_BATCH 5
done