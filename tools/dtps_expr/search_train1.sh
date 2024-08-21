port=$1
# baseline dtdec-oim_d5r2_sa-org_ca-res_ln_drop0.1
# sa expr
out_dir=dtdec-oim_d5r2_sa-linear_ca-res_ln_drop-0.1
python tools/train_dtps_baseline.py --config-file  \
configs/search/dtps/prw/dtps-prw_dtdec-oim_clip_d5r1.yaml \
--num-gpus 2 --resume --dist-url tcp://127.0.0.1:${port} \
SOLVER.IMS_PER_BATCH 8 REID_HEAD.SA linear REID_HEAD.NORM ln \
REID_HEAD.CA_RES True OUTPUT_DIR outputs/dtps/${out_dir}

out_dir=dtdec-oim_d5r2_sa-dynamic_ca-res_ln_drop-0.1
python tools/train_dtps_baseline.py --config-file  \
configs/search/dtps/prw/dtps-prw_dtdec-oim_clip_d5r1.yaml \
--num-gpus 2 --resume --dist-url tcp://127.0.0.1:${port} \
SOLVER.IMS_PER_BATCH 8 REID_HEAD.SA dynamic REID_HEAD.NORM ln \
REID_HEAD.CA_RES True OUTPUT_DIR outputs/dtps/${out_dir}

out_dir=dtdec-oim_d5r2_sa-resdynamic_ca-res_ln_drop-0.1
python tools/train_dtps_baseline.py --config-file  \
configs/search/dtps/prw/dtps-prw_dtdec-oim_clip_d5r1.yaml \
--num-gpus 2 --resume --dist-url tcp://127.0.0.1:${port} \
SOLVER.IMS_PER_BATCH 8 REID_HEAD.SA res_dynamic REID_HEAD.NORM ln \
REID_HEAD.CA_RES True OUTPUT_DIR outputs/dtps/${out_dir}

# drop expr
out_dir=dtdec-oim_d5r2_sa-org_ca-res_ln_drop-0.0
python tools/train_dtps_baseline.py --config-file  \
configs/search/dtps/prw/dtps-prw_dtdec-oim_clip_d5r1.yaml \
--num-gpus 2 --resume --dist-url tcp://127.0.0.1:${port} \
SOLVER.IMS_PER_BATCH 8 REID_HEAD.SA original REID_HEAD.NORM ln \
REID_HEAD.CA_RES True OUTPUT_DIR outputs/dtps/${out_dir} REID_HEAD.DROPOUT 0.0