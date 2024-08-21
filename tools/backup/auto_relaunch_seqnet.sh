count=0
cfg=$1
port=$2
nc=$3
while(( ${count}<6 ))
do
    let "count++"
    echo "Launch attemp "${count}" !"
    cd /home/patrick/Workspace/transearcher
    python tools/train_seqnet.py --config-file ${cfg}\
    --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port}
done
# for lvl in 4,3,2,1,0
# do
# python tools/train_ddetr.py --config-file ${cfg} \
# --num-gpus 4 --resume --dist-url tcp://127.0.0.1:${port} --eval-only MODEL.SEARCH.PERSON_FEAT.FEAT_BASE_LVL_IDX ${lvl}
# done