for hc in 0.05 0.1 0.2 0.4 0.6
do
    for hs in 0.05 0.1 0.3
    do
        python tools/train_seqnet_p.py --config-file configs/search/tips/ti_frcnn_C4Side_prw_1llr_16-8_12e_gem_dnboxa-10.yaml --num-gpus 1 \
        --resume --n-find-unparams --dist-url tcp://127.0.0.1:60666 SOLVER.MAX_ITER 27385 \
        REID_HEAD.BOX_AUGMENTATION.H_CENTER $hc REID_HEAD.BOX_AUGMENTATION.H_SCALE $hs \
        TEST.EVAL_PERIOD 2242 OUTPUT_DIR "outputs/tips/tircnn-c4side_16-8_prw_1llr_12e_gem_dnboxa-10-$hc-$hs"
    done
done
hs=0.2
for hc in 0.05 0.1 0.2 0.6
do
    python tools/train_seqnet_p.py --config-file configs/search/tips/ti_frcnn_C4Side_prw_1llr_16-8_12e_gem_dnboxa-10.yaml --num-gpus 1 \
    --resume --n-find-unparams --dist-url tcp://127.0.0.1:60666 SOLVER.MAX_ITER 27385 \
    REID_HEAD.BOX_AUGMENTATION.H_CENTER $hc REID_HEAD.BOX_AUGMENTATION.H_SCALE $hs \
    TEST.EVAL_PERIOD 2242 OUTPUT_DIR "outputs/tips/tircnn-c4side_16-8_prw_1llr_12e_gem_dnboxa-10-$hc-$hs"
done