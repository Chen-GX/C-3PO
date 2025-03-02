timestamp=$( date +"%Y%m%d_%H%M%S")

echo $timestamp

mkdir -p ./logs

n_plan_sample=2
checkpoint_dir=./model_cache/Qwen2-72B-Instruct

force_decision=False
force_action=Planning
filter_key='None'
focus_qid='None'

output_dir=./output_dir/run/batch_tree_search/${force_action}

for dataname in 2WikiMultiHopQA Musique hotpotqa NaturalQuestions PopQA TriviaQA
do
    echo "Running ${dataname}"
    # 获取checkpoint_dir的basename
    basename=$(basename ${checkpoint_dir})
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash offline_base_instruct.sh ${dataname} ${output_dir} ${n_plan_sample} ${checkpoint_dir} ${force_decision} ${force_action} ${filter_key} ${focus_qid} > ./logs/${force_action}_${dataname}_${basename}_${timestamp}.log 2>&1
    pkill -f /opt/conda/envs/c3po/bin/python
    sleep 300
done

wait