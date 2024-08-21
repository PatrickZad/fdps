<<!
for md in model_0038759.pth 
do
    python tools/train_spps_baseline.py --config-file configs/search/spps/def/prw/spps-prw_msroicoseg-oim.yaml \
    --num-gpus 4 --resume --eval-only --dist-url tcp://127.0.0.1:60666 SOLVER.IMS_PER_BATCH 8 \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    MODEL.WEIGHTS outputs_expd/downloads/def/spps-prw_msroicoseg-oim/${md} \
    TEST.IMS_PER_PROC 2 \
    OUTPUT_DIR outputs_expd/downloads/def/spps-prw_msroicoseg-oim \
    TEST.VIS True
done

out_dir=outputs/downloads/def/spps-cdps_msroifc-oim_b8_16e
for md in  model_final.pth
do
    python tools/train_spps_baseline.py --config-file configs/search/spps/def/cdps/spps-cdps_msroifc-oim_b8_16e.yaml \
    --num-gpus 4 --resume --eval-only --dist-url tcp://127.0.0.1:60666 SOLVER.IMS_PER_BATCH 8 \
    MODEL.WEIGHTS ${out_dir}/${md} \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    TEST.IMS_PER_PROC 2 \
    OUTPUT_DIR ${out_dir} \
    TEST.VIS True
    #REID_HEAD.VIS_INF True \
    #REID_HEAD.VIS_INF_SAVE ${out_dir}/vis_seg
done
out_dir=outputs/downloads/def/spps-cdps_msroicoseg-oim-triplet_apk_b8_16e
for md in  model_final.pth
do
    python tools/train_spps_baseline.py --config-file configs/search/spps/def/cdps/spps-cdps_msroicoseg-oim_b8_16e.yaml \
    --num-gpus 4 --resume --eval-only --dist-url tcp://127.0.0.1:60666 SOLVER.IMS_PER_BATCH 8 \
    MODEL.WEIGHTS ${out_dir}/${md} \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    TEST.IMS_PER_PROC 2 \
    OUTPUT_DIR ${out_dir} \
    TEST.VIS True
    #REID_HEAD.VIS_INF True \
    #REID_HEAD.VIS_INF_SAVE ${out_dir}/vis_seg
done
for md in model_0038759.pth 
do
    python tools/train_spps_baseline.py --config-file configs/search/spps/def/prw/spps-prw_msroifc-oim.yaml \
    --num-gpus 4 --resume --eval-only --dist-url tcp://127.0.0.1:60666 SOLVER.IMS_PER_BATCH 8 \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    MODEL.WEIGHTS outputs_expd/downloads/def/spps-prw_msroifc-oim/${md} \
    TEST.IMS_PER_PROC 2 \
    OUTPUT_DIR outputs_expd/downloads/def/spps-prw_msroifc-oim \
    TEST.VIS True
done
for md in model_0038759.pth 
do
    python tools/train_spps_baseline.py --config-file configs/search/spps/def/prw/spps-prw_msroicoseg-oim.yaml \
    --num-gpus 4 --resume --eval-only --dist-url tcp://127.0.0.1:60666 SOLVER.IMS_PER_BATCH 8 \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    MODEL.WEIGHTS outputs_expd/downloads/def/spps-prw_msroicoseg-oim-triplet_apk/${md} \
    TEST.IMS_PER_PROC 2 \
    OUTPUT_DIR outputs_expd/downloads/def/spps-prw_msroicoseg-oim-triplet_apk \
    TEST.VIS True
done
for md in model_0038759.pth 
do
    python tools/train_spps_baseline.py --config-file configs/search/spps/def/prw/spps-prw_msroifc-oim.yaml \
    --num-gpus 4 --resume --eval-only --dist-url tcp://127.0.0.1:60666 SOLVER.IMS_PER_BATCH 8 \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    MODEL.WEIGHTS outputs_expd/downloads/def/spps-prw_msroifc-oim-triplet/${md} \
    TEST.IMS_PER_PROC 2 \
    OUTPUT_DIR outputs_expd/downloads/def/spps-prw_msroifc-oim-triplet \
    TEST.VIS True
done
for md in model_0038759.pth 
do
    python tools/train_spps_baseline.py --config-file configs/search/spps/def/prw/spps-prw_msroifc-oim.yaml \
    --num-gpus 4 --resume --eval-only --dist-url tcp://127.0.0.1:60666 SOLVER.IMS_PER_BATCH 8 \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    MODEL.WEIGHTS outputs_expd/downloads/def/spps-prw_msroifc-oim-triplet_apk/${md} \
    TEST.IMS_PER_PROC 2 \
    OUTPUT_DIR outputs_expd/downloads/def/spps-prw_msroifc-oim-triplet_apk \
    TEST.VIS True
done

for md in model_0038759.pth 
do
    python tools/train_spps_baseline.py --config-file configs/search/spps/def/prw/spps-prw_msroicoseg-oim.yaml \
    --num-gpus 4 --resume --eval-only --dist-url tcp://127.0.0.1:60666 SOLVER.IMS_PER_BATCH 8 \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    MODEL.WEIGHTS outputs_expd/downloads/def/spps-prw_msroicoseg-oim-triplet/${md} \
    TEST.IMS_PER_PROC 2 \
    OUTPUT_DIR outputs_expd/downloads/def/spps-prw_msroicoseg-oim-triplet \
    TEST.VIS True
done
!
for md in model_0038759.pth 
do
    python tools/train_spps_baseline.py --config-file outputs_expd/refine/spps/def/spps-prw_msroifc-oim/config.yaml \
    --num-gpus 4 --resume --eval-only --dist-url tcp://127.0.0.1:60666 SOLVER.IMS_PER_BATCH 8 \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    MODEL.WEIGHTS outputs_expd/refine/spps/def/spps-prw_msroifc-oim/${md} \
    TEST.IMS_PER_PROC 2 \
    OUTPUT_DIR outputs_expd/refine/spps/def/spps-prw_msroifc-oim \
    TEST.VIS True
done
