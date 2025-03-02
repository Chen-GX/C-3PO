
mkdir -p ./logs

hostname -I

model_name_or_path=path_to_model_cache/Qwen2-72B-Instruct

TP=4
basename=$(basename $model_name_or_path)

bash base_sgl.sh $model_name_or_path $TP
