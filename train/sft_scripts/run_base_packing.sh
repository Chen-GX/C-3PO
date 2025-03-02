#!/bin/bash

GPU_NUM=$(nvidia-smi -L | wc -l)

seed=42

export TOKENIZERS_PARALLELISM=false
export VLLM_USE_MODELSCOPE="False"
export WANDB_DISABLED="True"

echo "Prepare the conda environment"

timestamp=$( date +"%Y%m%d_%H%M%S")
echo $timestamp

root_path=xxx
model_name_or_path=${root_path}/model_cache/Qwen2-0.5B
model_base_name=$(basename ${model_name_or_path})
dataset=policy_11.25_6data_v2
dataset_dir=${root_path}/proxy_train_data/sft

finetuning_type=full
learning_rate=4e-5

if [ $GPU_NUM -eq 8 ]; then
    gradient_accumulation_steps=6
    per_device_train_batch_size=8
elif [ $GPU_NUM -eq 4 ]; then
    gradient_accumulation_steps=12
    per_device_train_batch_size=8
else
    echo "GPU_NUM must be 4 or 8"
    exit 1
fi 


output_dir=${root_path}/workspace/output_dir/run/proxy_sft/${model_base_name}/${dataset}/${learning_rate}_${GPU_NUM}GPU_${timestamp}
deepspeed_config_file=${root_path}/workspace/LLaMA-Factory/examples/deepspeed/ds_z2_config.json
deepspeed_env=/opt/conda/envs/c3po/bin/deepspeed

${deepspeed_env} --num_gpus ${GPU_NUM} ../src/train.py \
    --deepspeed ${deepspeed_config_file} \
    --stage sft \
    --do_train \
    --flash_attn fa2 \
    --packing True \
    --neat_packing True \
    --model_name_or_path ${model_name_or_path} \
    --dataset_dir ${dataset_dir}\
    --dataset ${dataset} \
    --template qwen \
    --finetuning_type ${finetuning_type} \
    --save_safetensors \
    --output_dir ${output_dir} \
    --overwrite_cache \
    --max_new_tokens 2048 \
    --cutoff_len 4096 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 20 \
    --learning_rate ${learning_rate} \
    --num_train_epochs 8.0 \
    --dataloader_num_workers 16 \
    --preprocessing_num_workers 128 \
    --ddp_timeout 180000000 \
    --seed $seed \
    --plot_loss \
    --save_only_model \
    --bf16
