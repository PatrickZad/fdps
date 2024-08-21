count=0
while(( ${count}<6 ))
do
    python tools/train_seqnet.py --config-file configs/search/hoim/hoim_1c_cdps.yaml --num-gpus 1 --resume --dist-url tcp://127.0.0.1:60666
done
#seqnet/seqnet_1c_cdps.yaml