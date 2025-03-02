import json
import os.path as osp
import random
import importlib

from typing import List, Union

import logging

logger = logging.getLogger(__name__)

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_data(args):
    data_path = osp.join(args.datapath, args.dataname, args.data_file_name)
    data = load_jsonl(data_path)

    if args.filter:
        if args.focus_qid != "":
            focus_qid = load_json(osp.join(osp.dirname(args.filter_path), f"{args.focus_qid}.json"))
            if args.dataname in focus_qid:
                data_filter = focus_qid[args.dataname]
                data = [item for item in data if item['id'] in data_filter]
        else:
            filter_dict = load_json(args.filter_path)
            for key in ['no_retrieve']:
                if args.dataname not in filter_dict[key]:
                    continue
                data_filter = filter_dict[key][args.dataname]
                data = [item for item in data if item['id'] not in data_filter]
            filter_key = args.filter_key.split('@')
            for key in filter_key:
                if len(key) == 0:
                    continue
                if args.dataname in filter_dict[key]:
                    if args.proxy_model_name in filter_dict[key][args.dataname]:
                        data_filter = filter_dict[key][args.dataname][args.proxy_model_name]
                        data = [item for item in data if item['id'] not in data_filter]
        
    if args.debug_num > 0:
        data = data[:args.debug_num]
    
    return data

import socket

def is_port_in_use(port, host='0.0.0.0'):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False  # 端口未被占用
        except socket.error:
            return True  # 端口被占用

def load_agent(args, num_gpus=1):
    logger.info(f"GPU number: {num_gpus}")
    if args.backend == "vllm":
        logger.info(f"launching vllm engine")
        from vllm import LLM, SamplingParams
        llm = LLM(model=args.checkpoint_dir, tensor_parallel_size=num_gpus, seed=args.seed, trust_remote_code=True)
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            use_beam_search=False,
            max_tokens=args.max_tokens, 
            # best_of=args.n_generate_sample,
            # n=args.n_generate_sample,
            # stop=STOP,
        )
        return llm, sampling_params, None
    elif args.backend == "sglang":
        # logger.info(f"launching sglang engine")
        # import sglang as sgl
        # llm = sgl.Engine(model_path=args.checkpoint_dir, tp_size=num_gpus, random_seed=args.seed)
        sampling_params = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_tokens,
        }
        from sglang.utils import (
            execute_shell_command,
            wait_for_server,
        )
        while is_port_in_use(args.port):
            args.port += 1
        
        logger.info(f"launching sglang engine on port {args.port}")
        server_process = execute_shell_command(
            f"/opt/conda/envs/proxy_sg/bin/python -m sglang.launch_server --model-path {args.checkpoint_dir} --port {args.port} --host 0.0.0.0 --tp {args.tp} --dp {num_gpus // args.tp}"
        ) # you need to change the python env
        proxy_url = f"http://0.0.0.0:{args.port}"
        wait_for_server(proxy_url)
        from openai import OpenAI
        llm = OpenAI(
            api_key="EMPTY",
            base_url=f"{proxy_url}/v1",
        )

        return llm, sampling_params, server_process


def load_few_shot(args, config):
    examples = {}
    if args.test:
        few_shot_file_name = config['test_few_shot_file_name']
    else:
        few_shot_file_name = config['few_shot_file_name']
        
    for few_shot_type in few_shot_file_name:
        if few_shot_type in config['USE_PUBLIC'] and args.dataname in config['USE_PUBLIC'][few_shot_type]:
            import_path = f"prompt.few_shots.{few_shot_type}.public"
        else:
            import_path = f"prompt.few_shots.{few_shot_type}.{args.dataname}"

        module = importlib.import_module(import_path)
        few_shot_examples = getattr(module, "EXAMPLES", None)
        examples[few_shot_type] = few_shot_examples
    
    return examples


def few_shot_random_select(prompt_few_shots: Union[dict, List], action_type: str, num: int=3, dict_num: int=1, specified_key: str=None) -> List:
    prompt_few_shot = prompt_few_shots[action_type]
    if isinstance(prompt_few_shot, dict):
        if specified_key is None:
            # if example is a dict, randomly select an example from each key
            examples = []
            for key, value in prompt_few_shot.items():
                examples.extend(random.sample(value, k=min(dict_num, len(value))))
        else:
            # 分成dict和list两种情况
            if isinstance(prompt_few_shot[specified_key], list):
                examples = random.sample(prompt_few_shot[specified_key], k=min(num, len(prompt_few_shot[specified_key])))
            elif isinstance(prompt_few_shot[specified_key], dict):
                examples = []
                for key in prompt_few_shot[specified_key]:
                    examples.extend(random.sample(prompt_few_shot[specified_key][key], k=min(dict_num, len(prompt_few_shot[specified_key][key]))))
            else:
                raise ValueError("prompt_few_shot should be list or dict")

    elif isinstance(prompt_few_shot, list):
        examples = random.sample(prompt_few_shot, k=min(num, len(prompt_few_shots)))
        
    else:
        raise ValueError("prompt_few_shot should be list or dict")

    random.shuffle(examples)
    examples = [f"###\nExample {idx + 1}\n" + item for idx, item in enumerate(examples)]
    return "\n\n".join(examples)


def create_batches(data, batch_size):
    """将列表按照指定的 batch size 分成若干个子列表"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def print_tree(all_save_tree_info):
    from termcolor import colored 
    def print_docs(docs):
        line = []
        for idx, doc in enumerate(docs):
            line.append(f"Doc {idx + 1} ({doc['title']}): {doc['text']}")
        return "\n".join(line)
    
    for tree in all_save_tree_info:
        print(colored(f"Question: {tree['question']}", "red"))
        for node_id in tree['tree'].keys():
            node = tree['tree'][node_id]
            if node['role'] == 'MAKE_DECISION':
                print(colored(f"Role: {node['role']}", "red"))
                print(colored(f"{node['state']['output_text']}", "green"))
            elif node['role'] == 'RETRIEVE_AND_FILTER':
                print(colored(f"Role: {node['role']}", "red"))
                print(colored("Retrieved documents:", "blue"))
                print(colored(f"{print_docs(node['state']['retrieved_all_documents'])}\n\n", "blue"))
                print(colored(f"{node['state']['output_text']}", "green"))
            elif node['role'] == 'QUERY_LLM':
                print(colored(f"Role: {node['role']}", "red"))
                print(colored("Final documents: ", "blue"))
                print(colored(f"{print_docs(node['documents'])}\n\n", "blue"))
                print(colored(f"{node['state']['output_text']}", "green"))
            else:
                print(f"Role: {node['role']}")

        print("-" * 20)
