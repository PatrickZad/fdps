<<!
out_dir=outputs/downloads/def/spps-cdps_msroicoseg-oim-triplet_apk_b8_16e
for md in  model_final.pth
do
    python tools/train_spps_baseline.py --config-file configs/search/spps/def/cdps/spps-cdps_msroicoseg-oim_b8_16e.yaml \
    --num-gpus 4 --resume --eval-only --dist-url tcp://127.0.0.1:60666 SOLVER.IMS_PER_BATCH 8 \
    MODEL.WEIGHTS ${out_dir}/${md} \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    REID_HEAD.LOSS.METRIC.NAME triplet \
    REID_HEAD.PERSON_FEATURE.FEATURE_TYPE "global" \
    TEST.IMS_PER_PROC 2 \
    OUTPUT_DIR ${out_dir}
    #TEST.VIS True
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
    REID_HEAD.LOSS.METRIC.NAME triplet \
    REID_HEAD.PERSON_FEATURE.FEATURE_TYPE "local" \
    TEST.IMS_PER_PROC 2 \
    OUTPUT_DIR ${out_dir}
    #TEST.VIS True
    #REID_HEAD.VIS_INF True \
    #REID_HEAD.VIS_INF_SAVE ${out_dir}/vis_seg
done
out_dir=outputs_expd/refine/spps/def/spps-cdps_msroicoseg-oim-triplet_apk_b8_16e
for md in  model_final.pth
do
    python tools/train_spps_baseline.py --config-file configs/search/spps/def/cdps/spps-cdps_msroicoseg-oim_b8_16e.yaml \
    --num-gpus 2 --resume --eval-only --dist-url tcp://127.0.0.1:60666 SOLVER.IMS_PER_BATCH 8 \
    MODEL.WEIGHTS ${out_dir}/${md} \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    REID_HEAD.LOSS.METRIC.NAME triplet \
    REID_HEAD.PERSON_FEATURE.FEATURE_TYPE "cat" \
    TEST.IMS_PER_PROC 5 \
    OUTPUT_DIR ${out_dir}
    #TEST.VIS True
    #REID_HEAD.VIS_INF True \
    #REID_HEAD.VIS_INF_SAVE ${out_dir}/vis_seg
done
!
out_dir=outputs/refine/spps/def/spps-cdps_msroicoseg-oim-triplet_apk_09nlap
for md in model_0038759.pth 
do
    python tools/train_spps_baseline.py --config-file ${out_dir}/config.yaml \
    --num-gpus 2 --resume --eval-only --dist-url tcp://127.0.0.1:60666 SOLVER.IMS_PER_BATCH 10 \
    REID_HEAD.LOSS.METRIC.NAME triplet \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    MODEL.WEIGHTS ${out_dir}/${md} \
    TEST.IMS_PER_PROC 5 \
    OUTPUT_DIR ${out_dir}
    #REID_HEAD.VIS_INF True \
    #REID_HEAD.VIS_INF_SAVE ${out_dir}/vis_seg
done
<<!
out_dir=outputs/downloads/abl/def/spps-cdps_msroicoseg-oim-triplet_apk_6parts
for md in model_final.pth 
do
    python tools/train_spps_baseline.py --config-file configs/search/spps/def/cdps/spps-cdps_msroicoseg-oim_b8_16e.yaml \
    --num-gpus 4 --resume --eval-only --dist-url tcp://127.0.0.1:60666 SOLVER.IMS_PER_BATCH 8 \
    REID_HEAD.LOSS.METRIC.NAME triplet \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    REID_HEAD.PERSON_FEATURE.PART.NUM_PARTS 6 \
    MODEL.WEIGHTS ${out_dir}/${md} \
    TEST.IMS_PER_PROC 2 \
    #REID_HEAD.PERSON_FEATURE.FEATURE_TYPE "local" \
    OUTPUT_DIR ${out_dir}
    #REID_HEAD.VIS_INF True \
    #REID_HEAD.VIS_INF_SAVE ${out_dir}/vis_seg
done
out_dir=outputs/downloads/abl/def/spps-prw_msroicoseg-oim-triplet_apk_01nlap
for md in model_0038759.pth 
do
    python tools/train_spps_baseline.py --config-file configs/search/spps/def/prw/spps-prw_msroicoseg-oim.yaml \
    --num-gpus 4 --resume --eval-only --dist-url tcp://127.0.0.1:60666 SOLVER.IMS_PER_BATCH 10 \
    REID_HEAD.LOSS.METRIC.NAME triplet \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    #REID_HEAD.PERSON_FEATURE.PART.NUM_PARTS 7 \
    MODEL.WEIGHTS ${out_dir}/${md} \
    TEST.IMS_PER_PROC 2 \
    #REID_HEAD.PERSON_FEATURE.FEATURE_TYPE "cat" \
    OUTPUT_DIR ${out_dir}
    #REID_HEAD.VIS_INF True \
    #REID_HEAD.VIS_INF_SAVE ${out_dir}/vis_seg
done
out_dir=outputs/downloads/abl/def/spps-prw_msroicoseg-oim-triplet_apk_03nlap
for md in model_0038759.pth 
do
    python tools/train_spps_baseline.py --config-file configs/search/spps/def/prw/spps-prw_msroicoseg-oim.yaml \
    --num-gpus 4 --resume --eval-only --dist-url tcp://127.0.0.1:60666 SOLVER.IMS_PER_BATCH 10 \
    REID_HEAD.LOSS.METRIC.NAME triplet \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    #REID_HEAD.PERSON_FEATURE.PART.NUM_PARTS 7 \
    MODEL.WEIGHTS ${out_dir}/${md} \
    TEST.IMS_PER_PROC 2 \
    #REID_HEAD.PERSON_FEATURE.FEATURE_TYPE "cat" \
    OUTPUT_DIR ${out_dir}
    #REID_HEAD.VIS_INF True \
    #REID_HEAD.VIS_INF_SAVE ${out_dir}/vis_seg
done
out_dir=outputs/downloads/abl/def/spps-prw_msroicoseg-oim-triplet_apk_07nlap
for md in model_0038759.pth 
do
    python tools/train_spps_baseline.py --config-file configs/search/spps/def/prw/spps-prw_msroicoseg-oim.yaml \
    --num-gpus 4 --resume --eval-only --dist-url tcp://127.0.0.1:60666 SOLVER.IMS_PER_BATCH 10 \
    REID_HEAD.LOSS.METRIC.NAME triplet \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    #REID_HEAD.PERSON_FEATURE.PART.NUM_PARTS 7 \
    MODEL.WEIGHTS ${out_dir}/${md} \
    TEST.IMS_PER_PROC 2 \
    #REID_HEAD.PERSON_FEATURE.FEATURE_TYPE "cat" \
    OUTPUT_DIR ${out_dir}
    #REID_HEAD.VIS_INF True \
    #REID_HEAD.VIS_INF_SAVE ${out_dir}/vis_seg
done
out_dir=outputs/downloads/abl/def/spps-prw_msroicoseg-oim-triplet_apk_09nlap
for md in model_0038759.pth 
do
    python tools/train_spps_baseline.py --config-file configs/search/spps/def/prw/spps-prw_msroicoseg-oim.yaml \
    --num-gpus 4 --resume --eval-only --dist-url tcp://127.0.0.1:60666 SOLVER.IMS_PER_BATCH 10 \
    REID_HEAD.LOSS.METRIC.NAME triplet \
    REID_HEAD.LOSS.METRIC.NORM_FEAT True \
    #REID_HEAD.PERSON_FEATURE.PART.NUM_PARTS 7 \
    MODEL.WEIGHTS ${out_dir}/${md} \
    TEST.IMS_PER_PROC 2 \
    #REID_HEAD.PERSON_FEATURE.FEATURE_TYPE "cat" \
    OUTPUT_DIR ${out_dir}
    #REID_HEAD.VIS_INF True \
    #REID_HEAD.VIS_INF_SAVE ${out_dir}/vis_seg
done
#out_dir=outputs/downloads/def/spps-cuhk_msroicoseg-oim_1504
#for md in model_0078469.pth 
#do
#    python tools/train_spps_baseline.py --config-file configs/search/spps/def/ref/cuhk_1504.yaml \
#    --num-gpus 4 --resume --eval-only --dist-url tcp://127.0.0.1:60666 SOLVER.IMS_PER_BATCH 10 \
#    MODEL.WEIGHTS outputs/downloads/def/spps-cuhk_msroicoseg-oim_1504/${md} \
#    TEST.IMS_PER_PROC 2 \
#    OUTPUT_DIR outputs/downloads/def/spps-cuhk_msroicoseg-oim_1504 \
#    REID_HEAD.VIS_INF True \
#    REID_HEAD.VIS_INF_SAVE ${out_dir}/vis_seg
#done
!