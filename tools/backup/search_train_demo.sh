#! /bin/bash
for fwt in 1.0 0.8 0.6 0.4 0.2
do
    for gwt in 0.2 0.4 0.6 0.8 1.0
    do
        for g_level in 'coarse' 'fg' 'mid' 'hybrid'
        do
            python tools/train_net.py --config-file configs/cub/product_g.yaml --num-gpus 4 --resume \
            --dist-url tcp://127.0.0.1:50166 BENCHMARK.GATTENTION_LEVEL ${g_level} \
            BENCHMARK.F_WEIGHTS ${fwt} BENCHMARK.G_WEIGHTS ${gwt} \
            OUTPUT_DIR ./outputs/bmk/classify/cubg/product/${g_level}/f${fwt}g${gwt}
        done
    done
done