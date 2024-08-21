port=$1
nc=$2
count=0
while(( ${count}<3 ))
do
    let "count++"
    echo "Launch attemp "${count}" !"
    cd /home/patrick/Workspace/transearcher
    python tools/train_spps_baseline.py --config-file configs/search/spps/spps-prw_roifc-oim_newbs.yaml \
    --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 10 \
    SOLVER.MAX_ITER 41041 SOLVER.STEPS 29640,42000 \
    REID_HEAD.ID_ASSIGN.POS_SCORE_THRED 0.999 REID_HEAD.ID_ASSIGN.POS_IOU_THRED 0.001 \
    OUTPUT_DIR outputs/refine/spps/spps-prw_roifc-oim_ids10
done
count=8
while(( ${count}<3 ))
do
    let "count++"
    echo "Launch attemp "${count}" !"
    cd /home/patrick/Workspace/transearcher
    python tools/train_spps_baseline.py --config-file configs/search/spps/spps-prw_roifc-oim_newbs.yaml \
    --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 10 \
    SOLVER.MAX_ITER 41041 SOLVER.STEPS 29640,42000 \
    REID_HEAD.ID_ASSIGN.POS_SCORE_THRED 0.7 REID_HEAD.ID_ASSIGN.POS_IOU_THRED 0.2 \
    OUTPUT_DIR outputs/refine/spps/spps-prw_roifc-oim_ids0702
done
count=8
while(( ${count}<3 ))
do
    let "count++"
    echo "Launch attemp "${count}" !"
    cd /home/patrick/Workspace/transearcher
    python tools/train_spps_baseline.py --config-file configs/search/spps/spps-prw_roifc-oim_newbs.yaml \
    --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 10 \
    SOLVER.MAX_ITER 41041 SOLVER.STEPS 29640,42000 \
    REID_HEAD.ID_ASSIGN.POS_SCORE_THRED 0.7 REID_HEAD.ID_ASSIGN.POS_IOU_THRED 0.7 \
    OUTPUT_DIR outputs/refine/spps/spps-prw_roifc-oim_ids0707
done
count=8
while(( ${count}<3 ))
do
    let "count++"
    echo "Launch attemp "${count}" !"
    cd /home/patrick/Workspace/transearcher
    python tools/train_spps_baseline.py --config-file configs/search/spps/spps-prw_roifc-oim_newbs.yaml \
    --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 10 \
    SOLVER.MAX_ITER 41041 SOLVER.STEPS 29640,42000 \
    REID_HEAD.ID_ASSIGN.POS_SCORE_THRED 0.2 REID_HEAD.ID_ASSIGN.POS_IOU_THRED 0.7 \
    OUTPUT_DIR outputs/refine/spps/spps-prw_roifc-oim_ids0207
done

count=8
while(( ${count}<3 ))
do
    let "count++"
    echo "Launch attemp "${count}" !"
    cd /home/patrick/Workspace/transearcher
    python tools/train_spps_baseline.py --config-file configs/search/spps/spps-prw_roifc-oim_newbs.yaml \
    --num-gpus ${nc} --resume --dist-url tcp://127.0.0.1:${port} SOLVER.IMS_PER_BATCH 10 \
    SOLVER.MAX_ITER 41041 SOLVER.STEPS 29640,42000 \
    REID_HEAD.ID_ASSIGN.POS_SCORE_THRED 0.7 REID_HEAD.ID_ASSIGN.POS_IOU_THRED 0.5 \
    OUTPUT_DIR outputs/refine/spps/spps-prw_roifc-oim_ids0705
done