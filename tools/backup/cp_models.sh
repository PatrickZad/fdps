src_dir=$1
tgt_prt=$2
tgt_dir=${tgt_prt}/$(basename "${src_dir}")
mkdir ${tgt_dir}
cp ${src_dir}/config.yaml ${tgt_dir}/config.yaml
cp ${src_dir}/metrics.json ${tgt_dir}/metrics.json

if [[ ${src_dir} == *"cuhk"* ]]; then
    #cp ${src_dir}/model_0080711.pth ${tgt_dir}/model_0080711.pth
    cp ${src_dir}/model_0078469.pth ${tgt_dir}/model_0078469.pth
    #cp ${src_dir}/model_0076227.pth ${tgt_dir}/model_0076227.pth
    #cp ${src_dir}/model_0073985.pth ${tgt_dir}/model_0073985.pth
    #cp ${src_dir}/model_0069501.pth ${tgt_dir}/model_0069501.pth
    #cp ${src_dir}/model_0065017.pth ${tgt_dir}/model_0065017.pth
fi

if [[ ${src_dir} == *"prw"* ]]; then
    cp ${src_dir}/model_0038759.pth ${tgt_dir}/model_0038759.pth
fi

if [[ ${src_dir} == *"cdps"* ]]; then
    cp ${src_dir}/model_final.pth ${tgt_dir}/model_final.pth
fi