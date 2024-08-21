#! /bin/bash
for base in 2 5 0 1 3 4
do
    python tools/train_srcnn.py --config-file configs/search/srcnn/srcnn_prw_ss_1c_4lr.yaml --num-gpus 1 --resume \
    --dist-url tcp://127.0.0.1:56688 MODEL.SEARCH.PERSON_FEAT.FEAT_BASE_LVL_IDX ${base} \
    MODEL.SEARCH.PERSON_FEAT.ASSIGN_BASE_LVL_IDX ${base} \
    OUTPUT_DIR ./outputs/expand/srcnn/prw_ss_10_base${base}
done