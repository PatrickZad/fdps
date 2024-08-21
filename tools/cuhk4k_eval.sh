<<!
for expr_dir in  outputs/incmt/spps4/cuhk2prw2cdps_msroifc-oim_transfer_sslr \
outputs/incmt/spps4/cuhk2prw_msroifc-oim \
outputs/incmt/spps4/cuhk2prw_msroifc-oim_transfer \
outputs/incmt/spps4/cuhk2prw_msroifc-oim_transfer_slr \
outputs/incmt/spps4/cuhk2prw_msroifc-oim_transfer_sslr \
outputs/incmt/spps4/cuhk2prw_msroifc-oim_transfer_ssslr \
outputs/incmt/spps4/cuhk_msroifc-oim \
outputs/incmt/spps4/prw_msroifc-oim \
outputs/incmt/spps4/prw2cuhk_msroifc-oim \
outputs/incmt/spps4/prw2cuhk_msroifc-oim_transfer \
outputs/incmt/spps4/prw2cuhk_msroifc-oim_transfer_sslr
do 
    out_dir=${expr_dir}
    python tools/train_spps_baseline_cuhk4k-eval.py --config-file ${out_dir}/config.yaml \
    --num-gpus 2 --resume --eval-only --dist-url tcp://127.0.0.1:60888 SOLVER.IMS_PER_BATCH 10 \
    TEST.IMS_PER_PROC 5 \
    OUTPUT_DIR ${out_dir}
done

out_dir=outputs/incmt/spps4/cuhk2prw2cdps_msroifc-oim_transfer_sslr
python tools/train_spps_baseline.py --config-file ${out_dir}/config.yaml \
--num-gpus 2 --resume --eval-only --dist-url tcp://127.0.0.1:60888 SOLVER.IMS_PER_BATCH 10 \
TEST.IMS_PER_PROC 5 \
OUTPUT_DIR ${out_dir}
!
out_dir=outputs/refine/nova/spps-cdps_msroifc-oim-4head-gem
python tools/train_spps_baseline.py --config-file ${out_dir}/config.yaml \
--num-gpus 2 --resume --eval-only --dist-url tcp://127.0.0.1:60888 SOLVER.IMS_PER_BATCH 10 \
TEST.IMS_PER_PROC 5 \
OUTPUT_DIR ${out_dir}