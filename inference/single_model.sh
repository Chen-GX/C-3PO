#!/bin/bash

mkdir -p ./logs

dataname_list=2WikiMultiHopQA,hotpotqa,Musique,NaturalQuestions,PopQA,TriviaQA
# 按逗号分开
dataname_list=(${dataname_list//,/ })


# environment 
export VLLM_USE_MODELSCOPE="False"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

debug_num=-1
tp=1
proxy_concurrency=256
model_type=proxy
retriever_type=dense
retrieve_server_url=http://10.32.25.199:35004/search
musique_server_url=http://10.32.25.199:35002/search

llm_server_url=http://10.32.4.13:10080/v1,http://10.32.17.208:10080/v1  # your llm server

llm_name=qwen2-72b-instruct
llm_server_type=online

max_depth=13
test=True
temperature=0
online_concurrency=32
backend=sglang
output_dir=output_path
use_planning_cache=True
wo_llm=False
llm_query_few_shot=True

port=20010

folders=(
    path_to_your_ckpt
)

python=/opt/conda/envs/c3po/bin/python

for dataname in "${dataname_list[@]}"
do
    for folder in "${folders[@]}"
    do
        echo ${dataname}
        $python ../C-3PO/main.py \
            --model_type ${model_type} \
            --tp ${tp} \
            --proxy_concurrency ${proxy_concurrency} \
            --dataname ${dataname} \
            --output_dir ${output_dir} \
            --checkpoint_dir ${folder} \
            --retriever_type ${retriever_type} \
            --retrieve_server_url ${retrieve_server_url} \
            --musique_server_url ${musique_server_url} \
            --llm_name ${llm_name} \
            --llm_server_type ${llm_server_type} \
            --llm_server_url ${llm_server_url} \
            --debug_num ${debug_num} \
            --max_depth ${max_depth} \
            --test ${test} \
            --online_concurrency ${online_concurrency} \
            --temperature ${temperature} \
            --backend ${backend} \
            --use_planning_cache ${use_planning_cache} \
            --wo_llm ${wo_llm} \
            --llm_query_few_shot ${llm_query_few_shot} \
            --port ${port} > ./logs/${dataname}_ppo_215_105_ckpt150.log 2>&1
    done
done