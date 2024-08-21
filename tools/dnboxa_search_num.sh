for nbox in 2 4 6 8 12 14 16
do
    python tools/train_seqnet_p.py --config-file configs/search/tips/ti_frcnn_C4Side_prw_1llr_16-8_12e_gem_dnboxa-10.yaml --num-gpus 1 \
    --resume --n-find-unparams --dist-url tcp://127.0.0.1:60666 SOLVER.MAX_ITER 27385 \
    REID_HEAD.BOX_AUGMENTATION.NUM_LABELED $nbox REID_HEAD.BOX_AUGMENTATION.NUM_UNLABLED $nbox \
    TEST.EVAL_PERIOD 2242 OUTPUT_DIR "outputs/tips/tircnn-c4side_16-8_prw_1llr_12e_gem_dnboxa-$nbox"
done
