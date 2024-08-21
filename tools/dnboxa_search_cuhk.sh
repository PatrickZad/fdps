for hc in 0.05 0.1 0.2 0.4 0.6
do
    for hs in 0.05 0.1 0.3
    do
        python tools/train_seqnet_p.py --config-file configs/search/tips/ti_frcnn_C4Side_cuhk_1llr_16-8_10e_gem_dnboxa-6r.yaml --num-gpus 1 \
        --resume --n-find-unparams --dist-url tcp://127.0.0.1:60666 SOLVER.MAX_ITER 53809 \
        REID_HEAD.BOX_AUGMENTATION.H_CENTER $hc REID_HEAD.BOX_AUGMENTATION.H_SCALE $hs \
        TEST.EVAL_PERIOD 4484 OUTPUT_DIR "outputs/tips/tircnn-c4side_16-8_cuhk_1llr_10e_gem_dnboxa-6-$hc-$hs"
    done
done
hs=0.2
for hc in 0.05 0.1 0.2 0.6
do
    python tools/train_seqnet_p.py --config-file configs/search/tips/ti_frcnn_C4Side_cuhk_1llr_16-8_10e_gem_dnboxa-6r.yaml --num-gpus 1 \
    --resume --n-find-unparams --dist-url tcp://127.0.0.1:60666 SOLVER.MAX_ITER 53809 \
    REID_HEAD.BOX_AUGMENTATION.H_CENTER $hc REID_HEAD.BOX_AUGMENTATION.H_SCALE $hs \
    TEST.EVAL_PERIOD 4484 OUTPUT_DIR "outputs/tips/tircnn-c4side_16-8_cuhk_1llr_10e_gem_dnboxa-6-$hc-$hs"
done