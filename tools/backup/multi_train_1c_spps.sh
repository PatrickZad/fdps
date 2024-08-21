count=0
while(( ${count}<0 ))
do
    python tools/train_spps_baseline_apk.py --config-file configs/search/spps/def/cdps/spps-cdps_msroicoseg-oim_b4_24e.yaml \
    --num-gpus 1 --resume --dist-url tcp://127.0.0.1:60999 SOLVER.IMS_PER_BATCH 4 \
    REID_HEAD.LOSS.METRIC.NAME triplet \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    APK_DROP_LAST True \
    OUTPUT_DIR outputs/refine/spps/def/spps-cdps_msroicoseg-oim-triplet_apk_b4_24e_re
done
#seqnet/seqnet_1c_cdps.yaml
count=0
while(( ${count}<6 ))
do
    python tools/train_spps_baseline_apk.py --config-file configs/search/spps/def/cdps/spps-cdps_msroicoseg-oim_b4_24e.yaml \
    --num-gpus 1 --resume --dist-url tcp://127.0.0.1:60999 SOLVER.IMS_PER_BATCH 4 \
    REID_HEAD.LOSS.METRIC.NAME triplet \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    APK_DROP_LAST True \
    SOLVER.BASE_LR 0.00005 \
    OUTPUT_DIR outputs/refine/spps/def/spps-cdps_msroicoseg-oim-triplet_apk_b4_24e_re_lr
done