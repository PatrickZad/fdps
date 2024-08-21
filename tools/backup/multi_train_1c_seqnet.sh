count=0
while(( ${count}<2 ))
do
    python tools/train_seqnet.py --config-file configs/search/seqnet/seqnet_1c_cdps.yaml --num-gpus 1 --resume --dist-url tcp://127.0.0.1:60866 \
    OUTPUT_DIR outputs_expd/refine/spps/def/seqnet_cdps096
done
#seqnet/seqnet_1c_cdps.yaml