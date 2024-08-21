python tools/train_spps_baseline_apk.py --config-file configs/search/spps/def/cdps/spps-cdps_msroicoseg-oim_b8_16e.yaml \
--num-gpus 2 --resume --dist-url tcp://127.0.0.1:60888 \
--eval-only \
TEST.IMS_PER_PROC 5 \
SOLVER.IMS_PER_BATCH 8 \
REID_HEAD.LOSS.METRIC.NAME triplet \
REID_HEAD.LOSS.METRIC.NORM_FEAT True \
REID_HEAD.LOSS.LOSS_WEIGHTS.NON_LAP 0.1 \
APK_DROP_LAST True \
OUTPUT_DIR outputs/refine/spps/def/spps-cdps_msroicoseg-oim-triplet_apk_01nlap