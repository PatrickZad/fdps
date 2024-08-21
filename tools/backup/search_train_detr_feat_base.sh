#! /bin/bash
for base in 2 5 0 1 3 4
do
    python tools/train_ddetr.py --config-file configs/search/ddetr/4l_ib/ddetr_3c_prw_siou.yaml --num-gpus 3 --resume \
    --dist-url tcp://127.0.0.1:56789 MODEL.SEARCH.PERSON_FEAT.FEAT_BASE_LVL_IDX ${base} \
    MODEL.SEARCH.PERSON_FEAT.ASSIGN_BASE_LVL_IDX ${base} \
    OUTPUT_DIR ./outputs/expand/ddetr_4l_ib/prw_siou_10_base${base}_1lr
done