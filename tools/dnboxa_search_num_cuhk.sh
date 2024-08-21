for nbox in 10 12 14 16
do
    python tools/train_seqnet_relaunch.py --config-file configs/search/tips/ti_frcnn_C4Side_cuhk_1llr_16-8_10e_gem_dnboxa-8_b4.yaml --num-gpus 1 \
    --resume --n-find-unparams --dist-url tcp://127.0.0.1:60666 \
    REID_HEAD.BOX_AUGMENTATION.NUM_LABELED $nbox REID_HEAD.BOX_AUGMENTATION.NUM_UNLABLED $nbox \
    OUTPUT_DIR "outputs/tips/tircnn-c4side_16-8_cuhk_1llr_10e_gem_dnboxa-${nbox}_b4"
done
