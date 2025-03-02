import os, sys
os.chdir(sys.path[0])
import os.path as osp
import random
import yaml
import json
import logging
logger = logging.getLogger(__name__)

from tqdm import tqdm
from transformers import AutoTokenizer
from pebble import ProcessPool, ThreadPool

# from solver import Solver
# from retrieve.retriever import BasicRAG
# from search.search import Search
from arguments import get_args, set_seed
from utils import load_data, create_batches, load_agent, load_json
from utils import few_shot_random_select, load_few_shot
from prompt.planning import prompt_llm_planning, prompt_llm_planning_few_shot, instruct_for_each_dataset


class LLM_AGENT():
    def __init__(self, args, llm, sampling_params):
        self.args = args  
        self.llm = llm
        self.sampling_params = sampling_params
    
    def generate(self, query_list):
        index = query_list.get("index", 0)
        messages = query_list["messages"]
        sampling_params = query_list["sampling_params"]

        outputs = self.llm.chat.completions.create(
                    model="default",
                    messages=messages,
                    n=sampling_params['n'],
                    temperature=sampling_params['temperature'],
                    top_p=sampling_params['top_p'],
                    max_tokens=sampling_params['max_new_tokens'],
                )
        return {"index": index, "outputs": outputs}

if __name__=="__main__":
    args = get_args(write_to_file=False)
    set_seed(args.seed)

    if args.test:
        args.cache_dir = osp.join(args.cache_dir, args.dataname, "test.json")
    else:
        args.cache_dir = osp.join(args.cache_dir, args.dataname, "ppo_train.json")

    args.max_tokens = 2048
    args.temperature = 0.3
    args.top_k = -1
    args.top_p = 1.0
    args.model_type = "proxy"
    args.n_plan_sample = 5
    args.filter = False
    args.few_shot = True
    args.focus_qid = ""

    args.data_file_name = "ppo_train_6000.jsonl"

    # tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)

    if osp.exists(args.cache_dir):
        cache_data = load_json(args.cache_dir)
    else:
        os.makedirs(osp.dirname(args.cache_dir), exist_ok=True)
        cache_data = {}

    data = load_data(args)
    logger.info(f"{args.dataname} data loaded, length: {len(data)}")

    available_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(',')
    print(f"available_gpus: {available_gpus}")
    llm, sampling_params, server_process = load_agent(args, num_gpus=len(available_gpus))
    llm_agent = LLM_AGENT(args, llm, sampling_params)

    need_generate_data = []
    for item in data:
        key = f"{item['id']}_{item['question']}"
        if key not in cache_data:
            need_generate_data.append(item)

    with open(args.search_config, 'r') as f:
        config = yaml.safe_load(f)
    few_shot_examples = load_few_shot(args, config=config)

    batch_size = 1000
    
    for i in tqdm(range(0, len(need_generate_data), batch_size)):
        batch_data = need_generate_data[i:i + batch_size]
        key_list = []
        prompt_list = []
        
        # 对每个batch内的数据生成prompt
        for item in batch_data:
            key = f"{item['id']}_{item['question']}"
            for i in range(args.n_plan_sample):
                key_list.append(key)
                info = {
                    "examples": few_shot_random_select(few_shot_examples, 'planning', num=args.few_num, dict_num=args.dict_few_num),
                    "question": item['question'],
                    "dataset_instructions": instruct_for_each_dataset.get(args.dataname, ""),
                }
                input_text = prompt_llm_planning_few_shot.format_map(info)
                message = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input_text},
                ]
                # input_template_text = tokenizer.apply_chat_template(
                #     message,
                #     tokenize=False,
                #     add_generation_prompt=True
                # )
                prompt_list.append(message)
        
        # 处理当前batch
        sampling_params["n"] = 2
        query_list = [{"index": i, "messages": p, "sampling_params": sampling_params} for i, p in enumerate(prompt_list)]
        with ThreadPool(max_workers=args.proxy_concurrency) as pool:
            future = pool.map(llm_agent.generate, query_list, timeout=180)
            outputs = list(future.result())
        # outputs = llm.generate(prompt_list, sampling_params)
        # outputs = [outputs[i:i+sampling_params["n"]] for i in range(0, len(outputs), sampling_params["n"])]
        
        # 保存结果
        for key, output in zip(key_list, outputs):
            if key not in cache_data:
                cache_data[key] = []
            for o in output['outputs'].choices:
                if o.message.content not in cache_data[key]:
                    cache_data[key].append(o.message.content)


    print(f"cache_data length: {len(cache_data)}")
    print(f"save at {args.cache_dir}")
    print(f"per question generate {args.n_plan_sample * sampling_params['n']} samples")
    with open(args.cache_dir, 'w') as f:
        json.dump(cache_data, f, indent=2)

    from sglang.utils import terminate_process
    terminate_process(server_process)