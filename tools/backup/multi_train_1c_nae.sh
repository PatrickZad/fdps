count=0
while(( ${count}<2 ))
do
    python tools/train_seqnet.py --config-file configs/search/nae/nae_1c_cdps.yaml --num-gpus 1 --resume --dist-url tcp://127.0.0.1:60886 \
    OUTPUT_DIR outputs_expd/refine/spps/def/nae_cdps096
done
#seqnet/seqnet_1c_cdps.yaml