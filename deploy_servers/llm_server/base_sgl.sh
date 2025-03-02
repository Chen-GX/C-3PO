model_name_or_path=$1
TP=$2

export VLLM_USE_MODELSCOPE="False"

python=path_to/c3po/bin/python

# 自动检测gpu数量
GPU_NUM=$(nvidia-smi -L | wc -l)

${python} -m sglang.launch_server --model-path $model_name_or_path --host 0.0.0.0 --tp $TP --dp $((${GPU_NUM} / ${TP})) --port 10080 --mem-fraction-static 0.9

