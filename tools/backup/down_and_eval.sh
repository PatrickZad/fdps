# scp -r patrick@192.168.30.26:/home/patrick/Workspace/transearcher/outputs/refine/spps/prev/spps-prw_msroicoseg-oim-triplet-norm_apk_cl outputs/downloads/prev/

#for md in model_0041039.pth model_0038759.pth # model_0036479.pth model_final.pth
#do
#    python tools/train_spps_baseline.py --config-file configs/search/spps/def/prw/spps-prw_msroifc-oim.yaml \
#    --num-gpus 4 --resume --eval-only --dist-url tcp://127.0.0.1:60666 SOLVER.IMS_PER_BATCH 8 \
#    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
#    MODEL.WEIGHTS outputs_expd/downloads/def/spps-prw_msroifc-oim/${md} \
#    TEST.IMS_PER_PROC 2 \
#    OUTPUT_DIR outputs_expd/downloads/def/spps-prw_msroifc-oim
#done
#out_dir=outputs_expd/downloads/def/spps-cuhk_msroicoseg-oim-triplet_apk_droplast_500
#for md in  model_0080711.pth model_0076227.pth model_0078469.pth
#do
#    python tools/train_spps_baseline.py --config-file configs/search/spps/def/ref/cuhk_500.yaml \
#    --num-gpus 4 --resume --eval-only --dist-url tcp://127.0.0.1:60666 SOLVER.IMS_PER_BATCH 8 \
#    MODEL.WEIGHTS ${out_dir}/${md} \
#    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
#    TEST.IMS_PER_PROC 2 \
#    OUTPUT_DIR ${out_dir}
#done
out_dir=outputs/downloads/def/spps-cdps_msroicoseg-oim_b8_16e
for md in  model_final.pth model_0057855.pth model_0054239.pth
do
    python tools/train_spps_baseline.py --config-file configs/search/spps/def/cdps/spps-cdps_msroicoseg-oim_b8_16e.yaml \
    --num-gpus 4 --resume --eval-only --dist-url tcp://127.0.0.1:60666 SOLVER.IMS_PER_BATCH 8 \
    MODEL.WEIGHTS ${out_dir}/${md} \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    TEST.IMS_PER_PROC 2 \
    OUTPUT_DIR ${out_dir}
done
out_dir=outputs/downloads/def/spps-cdps_msroicoseg-oim-triplet
for md in  model_final.pth model_0057855.pth model_0054239.pth
do
    python tools/train_spps_baseline.py --config-file configs/search/spps/def/cdps/spps-cdps_msroicoseg-oim_b8_16e.yaml \
    --num-gpus 4 --resume --eval-only --dist-url tcp://127.0.0.1:60666 SOLVER.IMS_PER_BATCH 8 \
    MODEL.WEIGHTS ${out_dir}/${md} \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    TEST.IMS_PER_PROC 2 \
    OUTPUT_DIR ${out_dir}
done
out_dir=outputs/downloads/def/spps-cdps_msroicoseg-oim-triplet_apk_b8_16e
for md in  model_final.pth model_0057855.pth model_0054239.pth
do
    python tools/train_spps_baseline.py --config-file configs/search/spps/def/cdps/spps-cdps_msroicoseg-oim_b8_16e.yaml \
    --num-gpus 4 --resume --eval-only --dist-url tcp://127.0.0.1:60666 SOLVER.IMS_PER_BATCH 8 \
    MODEL.WEIGHTS ${out_dir}/${md} \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    TEST.IMS_PER_PROC 2 \
    OUTPUT_DIR ${out_dir}
done
out_dir=outputs/downloads/def/spps-cdps_msroifc-oim_b8_16e
for md in  model_final.pth model_0057855.pth model_0054239.pth
do
    python tools/train_spps_baseline.py --config-file configs/search/spps/def/cdps/spps-cdps_msroifc-oim_b8_16e.yaml \
    --num-gpus 4 --resume --eval-only --dist-url tcp://127.0.0.1:60666 SOLVER.IMS_PER_BATCH 8 \
    MODEL.WEIGHTS ${out_dir}/${md} \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    TEST.IMS_PER_PROC 2 \
    OUTPUT_DIR ${out_dir}
done
out_dir=outputs/downloads/def/spps-cdps_msroifc-oim-triplet
for md in  model_final.pth model_0057855.pth model_0054239.pth
do
    python tools/train_spps_baseline.py --config-file configs/search/spps/def/cdps/spps-cdps_msroifc-oim_b8_16e.yaml \
    --num-gpus 4 --resume --eval-only --dist-url tcp://127.0.0.1:60666 SOLVER.IMS_PER_BATCH 8 \
    MODEL.WEIGHTS ${out_dir}/${md} \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    TEST.IMS_PER_PROC 2 \
    OUTPUT_DIR ${out_dir}
done
#out_dir=outputs/downloads/def/spps-cdps_msroifc-oim-triplet_apk_b8_16e_re
#for md in  model_final.pth model_0057855.pth model_0054239.pth
#do
#    python tools/train_spps_baseline.py --config-file configs/search/spps/def/cdps/spps-cdps_msroifc-oim_b8_16e.yaml \
#    --num-gpus 4 --resume --eval-only --dist-url tcp://127.0.0.1:60666 SOLVER.IMS_PER_BATCH 8 \
#    MODEL.WEIGHTS ${out_dir}/${md} \
#    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
#    TEST.IMS_PER_PROC 2 \
#    OUTPUT_DIR ${out_dir}
#done