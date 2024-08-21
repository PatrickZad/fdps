out_dir=outputs/refine/downloads/vis/spps-prw_msroicoseg-oim_mstp
for md in model_0054719.pth 
do
    python tools/train_spps_baseline.py --config-file ${out_dir}/config.yaml \
    --num-gpus 2 --resume --eval-only --dist-url tcp://127.0.0.1:60888 SOLVER.IMS_PER_BATCH 10 \
    MODEL.WEIGHTS ${out_dir}/${md} \
    TEST.IMS_PER_PROC 5 \
    OUTPUT_DIR ${out_dir} \
    REID_HEAD.VIS_INF True \
    REID_HEAD.VIS_INF_SAVE ${out_dir}/vis_seg
done

<<<<<<< HEAD
=======
out_dir=outputs/refine/downloads/vis/spps-cuhk_msroicoseg-oim-triplet-norm_apk
for md in model_0078469.pth 
do
    python tools/train_spps_baseline.py --config-file ${out_dir}/config.yaml \
    --num-gpus 2 --resume --eval-only --dist-url tcp://127.0.0.1:60888 SOLVER.IMS_PER_BATCH 10 \
    MODEL.WEIGHTS ${out_dir}/${md} \
    TEST.IMS_PER_PROC 5 \
    OUTPUT_DIR ${out_dir} \
    REID_HEAD.VIS_INF True \
    REID_HEAD.VIS_INF_SAVE ${out_dir}/vis_seg
done
#prw
for dir_post in mstp
do
    out_dir=outputs/refine/spps/spps-prw_msroicoseg-oim_${dir_post}
    python tools/train_spps_baseline.py --config-file ${out_dir}/config.yaml \
    --num-gpus 2 --resume --eval-only --dist-url tcp://127.0.0.1:60888 SOLVER.IMS_PER_BATCH 10 \
    MODEL.WEIGHTS ${out_dir}/model_0038759.pth \
    TEST.IMS_PER_PROC 5 \
    OUTPUT_DIR ${out_dir} \
    REID_HEAD.VIS_INF True \
    REID_HEAD.VIS_INF_SAVE ${out_dir}/vis_seg
done
#cuhk
for dir_post in roicoseg-oim_new msroicoseg-oim_new
do
    out_dir=outputs/refine/spps/spps-cuhk_${dir_post}
    python tools/train_spps_baseline.py --config-file ${out_dir}/config.yaml \
    --num-gpus 2 --resume --eval-only --dist-url tcp://127.0.0.1:60888 SOLVER.IMS_PER_BATCH 10 \
    MODEL.WEIGHTS ${out_dir}/model_0038759.pth \
    TEST.IMS_PER_PROC 5 \
    OUTPUT_DIR ${out_dir} \
    REID_HEAD.VIS_INF True \
    REID_HEAD.VIS_INF_SAVE ${out_dir}/vis_seg
done
#cdps
for dir_post in triplet_apk_03nlap triplet_apk_07nlap
do
    out_dir=outputs/refine/spps/def/spps-cdps_msroicoseg-oim-${dir_post}
    python tools/train_spps_baseline.py --config-file ${out_dir}/config.yaml \
    --num-gpus 2 --resume --eval-only --dist-url tcp://127.0.0.1:60888 SOLVER.IMS_PER_BATCH 10 \
    MODEL.WEIGHTS ${out_dir}/model_0038759.pth \
    TEST.IMS_PER_PROC 5 \
    OUTPUT_DIR ${out_dir} \
    REID_HEAD.VIS_INF True \
    REID_HEAD.VIS_INF_SAVE ${out_dir}/vis_seg
done
!
out_dir=outputs/incmt/spps4/prw_msroifc-oim
python tools/train_spps_baseline.py --config-file ${out_dir}/config.yaml \
--num-gpus 2 --resume --eval-only --dist-url tcp://127.0.0.1:60888 SOLVER.IMS_PER_BATCH 10 \
MODEL.WEIGHTS ${out_dir}/model_0038759.pth \
TEST.IMS_PER_PROC 5 \
OUTPUT_DIR ${out_dir}
>>>>>>> c4ed0dc31e8a648441f41686c793b150c62ab77a
