for mn in 24661 26903 29145 31387 33629 35871 38113 40355 42597 44839 47081 49323 51565 53807 56049 58291 60533 62775 65017 67259
do
    echo "model_00${mn}.pth" > "outputs/tips/tiretina-c4side_16-8_cuhk_1llr_10e_gem_dnboxa-9_botinit/last_checkpoint"
    python tools/train_retina_baseline_p.py --config-file configs/search/tips/ti_retina_C4Side_cuhk_1llr_16-8_10e_gem_dnboxa-9_botinit.yaml --num-gpus 1 --resume --n-find-unparams --dist-url tcp://127.0.0.1:60666 --eval-only
done