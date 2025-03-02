
dataname=$1
output_dir=$2

export VLLM_USE_MODELSCOPE="False"
python=/opt/conda/envs/c3po/bin/python

model_type=proxy

# parameters
n_decision_sample=3
n_generate_sample=2
n_plan_sample=$3
checkpoint_dir=$4

retrieve_server_url=http://10.32.25.199:35004/search
musique_server_url=http://10.32.25.199:35002/search
llm_server_type=offline

filter=False
filter_path=path/sampling_filter.json

force_decision=${5:-False}
force_action=${6:-Planning}
filter_key=${7:-existing}
focus_qid=${8:-""}

max_depth=13

debug_num=-1
seed=0
test=False
backend=sglang

${python} ../C-3PO/main.py \
    --model_type ${model_type} \
    --dataname ${dataname} \
    --output_dir ${output_dir} \
    --n_decision_sample ${n_decision_sample} \
    --n_plan_sample ${n_plan_sample} \
    --n_generate_sample ${n_generate_sample} \
    --checkpoint_dir ${checkpoint_dir} \
    --retrieve_server_url ${retrieve_server_url} \
    --musique_server_url ${musique_server_url} \
    --llm_server_type ${llm_server_type} \
    --debug_num ${debug_num} \
    --filter ${filter} \
    --filter_path ${filter_path} \
    --force_decision ${force_decision} \
    --force_action ${force_action} \
    --filter_key ${filter_key} \
    --max_depth ${max_depth} \
    --focus_qid ${focus_qid} \
    --seed ${seed} \
    --test ${test} \
    --backend ${backend}