src_dir=$1
if [[ ${src_dir} == *"spps"* ]]; then
    if [[ ${src_dir} == *"cuhk"* ]]; then
        rm ${src_dir}/model_000*
        rm ${src_dir}/model_001*
        rm ${src_dir}/model_002*
        rm ${src_dir}/model_003*
        rm ${src_dir}/model_004*
        rm ${src_dir}/model_005*
        rm ${src_dir}/model_006*
    fi

    if [[ ${src_dir} == *"prw"* ]]; then
        rm ${src_dir}/model_000*
        rm ${src_dir}/model_001*
        rm ${src_dir}/model_002*
    fi

    if [[ ${src_dir} == *"cdps"* ]]; then
        rm ${src_dir}/model_000*
        rm ${src_dir}/model_001*
        rm ${src_dir}/model_002*
        rm ${src_dir}/model_003*
        rm ${src_dir}/model_004*
    fi
fi
if [[ ${src_dir} == *"dtdec"* ]]; then
    if [[ ${src_dir} == *"cuhk"* ]]; then
        rm ${src_dir}/model_000*
        rm ${src_dir}/model_001*
        rm ${src_dir}/model_002*
        rm ${src_dir}/model_003*
        rm ${src_dir}/model_004*
        rm ${src_dir}/model_005*
        rm ${src_dir}/model_006*
        rm ${src_dir}/model_007*
    fi

    if [[ ${src_dir} == *"prw"* ]]; then
        rm ${src_dir}/model_000*
        rm ${src_dir}/model_001*
        rm ${src_dir}/model_002*
        rm ${src_dir}/model_002*
    fi

    if [[ ${src_dir} == *"cdps"* ]]; then
        rm ${src_dir}/model_000*
        rm ${src_dir}/model_001*
        rm ${src_dir}/model_002*
        rm ${src_dir}/model_003*
        rm ${src_dir}/model_004*
    fi
fi