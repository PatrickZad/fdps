for ft in "global" "local" "cat"
do
    python tools/train_spps_baseline.py --config-file configs/search/spps/spps-prw_msroicoseg-oim-triplet.yaml \
    --num-gpus 1 --resume --dist-url tcp://127.0.0.1:59999 --eval-only \
    INPUT.SIZE_DIVISIBILITY 1 \
    REID_HEAD.PERSON_FEATURE.FEATURE_TYPE ${ft} \
    CWS True \
    MODEL.WEIGHTS "outputs/refine/spps/spps-prw_msroicoseg-oim-triplet-norm/model_0036479.pth" \
    OUTPUT_DIR outputs/refine/spps/eval/spps-prw_msroicoseg-oim-triplet-norm_cws

    python tools/train_spps_baseline.py --config-file configs/search/spps/spps-prw_msroicoseg-oim-triplet.yaml \
    --num-gpus 1 --resume --dist-url tcp://127.0.0.1:59999 --eval-only \
    INPUT.SIZE_DIVISIBILITY 1 \
    REID_HEAD.PERSON_FEATURE.FEATURE_TYPE ${ft} \
    CWS True \
    MODEL.WEIGHTS "outputs/refine/spps/spps-prw_msroicoseg-oim-triplet-norm_apk/model_0036479.pth" \
    OUTPUT_DIR outputs/refine/spps/eval/spps-prw_msroicoseg-oim-triplet-norm-apk_cws
    
    python tools/train_spps_baseline.py --config-file configs/search/spps/spps-cuhk_msroicoseg-oim-triplet.yaml \
    --num-gpus 1 --resume --dist-url tcp://127.0.0.1:59999 --eval-only \
    INPUT.SIZE_DIVISIBILITY 1 \
    REID_HEAD.PERSON_FEATURE.FEATURE_TYPE ${ft} \
    CWS True \
    MODEL.WEIGHTS "outputs/refine/spps/spps-cuhk_msroicoseg-oim-triplet/model_0073985.pth" \
    OUTPUT_DIR outputs/refine/spps/eval/spps-cuhk_msroicoseg-oim-triplet_cws

    python tools/train_spps_baseline.py --config-file configs/search/spps/spps-cuhk_msroicoseg-oim-triplet.yaml \
    --num-gpus 1 --resume --dist-url tcp://127.0.0.1:59999 --eval-only \
    INPUT.SIZE_DIVISIBILITY 1 \
    REID_HEAD.PERSON_FEATURE.FEATURE_TYPE ${ft} \
    CWS True \
    MODEL.WEIGHTS "outputs/refine/spps/spps-cuhk_msroicoseg-oim-triplet_apk/model_0076227.pth" \
    OUTPUT_DIR outputs/refine/spps/eval/spps-cuhk_msroicoseg-oim-triplet-apk_cws

    python tools/train_spps_baseline.py --config-file configs/search/spps/spps-cuhk_msroicoseg-oim-triplet.yaml \
    --num-gpus 1 --resume --dist-url tcp://127.0.0.1:59999 --eval-only \
    INPUT.SIZE_DIVISIBILITY 1 \
    REID_HEAD.PERSON_FEATURE.FEATURE_TYPE ${ft} \
    CWS True \
    MODEL.WEIGHTS "outputs/refine/spps/spps-cuhk_msroicoseg-oim-triplet-norm/model_0071743.pth" \
    OUTPUT_DIR outputs/refine/spps/eval/spps-cuhk_msroicoseg-oim-triplet-norm_cws

    python tools/train_spps_baseline.py --config-file configs/search/spps/spps-cuhk_msroicoseg-oim-triplet.yaml \
    --num-gpus 1 --resume --dist-url tcp://127.0.0.1:59999 --eval-only \
    INPUT.SIZE_DIVISIBILITY 1 \
    REID_HEAD.PERSON_FEATURE.FEATURE_TYPE ${ft} \
    CWS True \
    MODEL.WEIGHTS "outputs/refine/spps/spps-cuhk_msroicoseg-oim-triplet-norm_apk/model_0069501.pth" \
    OUTPUT_DIR outputs/refine/spps/eval/spps-cuhk_msroicoseg-oim-triplet-norm-apk_cws
done