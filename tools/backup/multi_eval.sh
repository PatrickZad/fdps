port=$1
nc=$2
#CUHK
prefix=outputs/refine/spps/def/spps-cuhk_msroicoseg-oim
for ref in 1504 gap nclip sscale syncbn 
do
    out_dir=${prefix}_${ref}
    mv ${out_dir}/last_checkpoint ${out_dir}/bk_last_checkpoint
    for weight in model_final.pth model_0080711.pth model_0078469.pth \
    model_0073985.pth model_0069501.pth model_0065017.pth
    do
        python tools/train_spps_baseline.py --config-file configs/search/spps/def/ref/cuhk_${ref}.yaml \
        --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} \
        --eval-only MODEL.WEIGHTS ${out_dir}/${weight} \
        OUTPUT_DIR ${out_dir}
    done
    mv ${out_dir}/bk_last_checkpoint ${out_dir}/last_checkpoint
done
prefix=outputs_expd/refine/spps/def/spps-cuhk_msroicoseg-oim
for ref in triplet triplet_apk triplet_apk_droplast
do
    out_dir=${prefix}-${ref}
    mv ${out_dir}/last_checkpoint ${out_dir}/bk_last_checkpoint
    for weight in model_final.pth model_0080711.pth model_0078469.pth \
    model_0073985.pth model_0069501.pth model_0065017.pth
    do
        python tools/train_spps_baseline.py --config-file configs/search/spps/def/cuhk/spps-cuhk_msroicoseg-oim.yaml \
        --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} \
        --eval-only MODEL.WEIGHTS ${out_dir}/${weight} \
        OUTPUT_DIR ${out_dir}
    done
    mv ${out_dir}/bk_last_checkpoint ${out_dir}/last_checkpoint
done

out_dir=outputs/refine/spps/def/spps-cuhk_msroicoseg-oim
mv ${out_dir}/last_checkpoint ${out_dir}/bk_last_checkpoint
for weight in model_final.pth model_0080711.pth model_0078469.pth \
model_0073985.pth model_0069501.pth model_0065017.pth
do
    python tools/train_spps_baseline.py --config-file configs/search/spps/def/cuhk/spps-cuhk_msroicoseg-oim.yaml \
    --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} \
    --eval-only MODEL.WEIGHTS ${out_dir}/${weight} \
    OUTPUT_DIR ${out_dir}
done
mv ${out_dir}/bk_last_checkpoint ${out_dir}/last_checkpoint
#PRW

#CDPS