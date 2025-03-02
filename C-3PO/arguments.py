import os
import os.path as osp
import time
import argparse
import random
import numpy as np
import torch
from log_utils import log_params

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        assert False, f"Unsupported value: {v}"

def set_seed(seed: int = 1024) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


data_file_name = {
    "2WikiMultiHopQA": "train_6000.jsonl",
    "NaturalQuestions": "train_6000.jsonl",
    "PopQA": "train_6000.jsonl",
    "TriviaQA": "train_6000.jsonl",
    "hotpotqa": "train_6000.jsonl",
    "Musique": "train_6000.jsonl",
}

def get_args(write_to_file=True):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)

    ## params of LLM
    parser.add_argument('--checkpoint_dir', type=str, default='path_to_ckpt')
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--port', type=int, default=20000)
    parser.add_argument('--proxy_concurrency', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument('--model_type', type=str, default='proxy', choices=['proxy', 'gpt'])
    parser.add_argument('--gpt_proxy_concurrency', type=int, default=16)
    parser.add_argument('--openai_api_key', type=str, default="")
    parser.add_argument('--proxy_url', type=str, default="")
    parser.add_argument('--proxy_model_name', type=str, default="gpt-4o-mini")
    parser.add_argument('--force_decision', type=str2bool, default=False)
    parser.add_argument('--force_action', type=str, default="Retrieval", choices=["Planning", "Retrieval", "No Retrieval"])

    ## params of search
    parser.add_argument('--n_decision_sample', type=int, default=3, help="how many samples generated for decision step")
    parser.add_argument('--n_generate_sample', type=int, default=2, help="how many samples generated for each step")
    parser.add_argument('--n_plan_sample', type=int, default=2, help="how many samples generated for plan step")
    parser.add_argument('--n_answer_sample', type=int, default=1, help="how many samples generated for answer step") 
    parser.add_argument('--max_iter', type=int, default=20, help="maximally allowed iterations")
    parser.add_argument('--search_config', type=str, default="./config/search_config.yaml")
    parser.add_argument('--few_shot', type=str2bool, default=True)
    parser.add_argument('--few_num', type=int, default=2)
    parser.add_argument('--dict_few_num', type=int, default=1)
    parser.add_argument('--decide_prompt', type=str, default="identity")
    parser.add_argument('--max_depth', type=int, default=9, help="maximum step of solution")
    parser.add_argument('--max_documents', type=int, default=10, help="maximum documents")
    parser.add_argument('--answer_eval', type=str, default="compare", choices=["extract", "compare", "none"])

    ## params of retrieve
    parser.add_argument('--retriever_type', type=str, default="dense", choices=["dense", "search_engine"])
    parser.add_argument('--retrieve_server_url', type=str, default="http://10.32.25.199:35004/search")
    parser.add_argument('--musique_server_url', type=str, default="http://10.32.25.199:35002/search")
    parser.add_argument('--retrieve_top_k', type=int, default=10)
    parser.add_argument('--max_query_length', type=int, default=100)
    # params of search engine
    parser.add_argument('--search_engine_url', type=str, default="")
    parser.add_argument('--search_scene', type=str, default="")
    parser.add_argument('--search_engine_cache', type=str2bool, default=True)
    parser.add_argument('--search_engine_cache_file', type=str, default="path ot cache_search")
    
    ## params of llm
    parser.add_argument('--llm_api_key', type=str, default="EMPTY")
    parser.add_argument('--llm_server_url', type=str, default="http://10.32.32.13:10080/v1")
    parser.add_argument('--llm_name', type=str, default="qwen2-72b-instruct")
    parser.add_argument('--llm_server_type', type=str, default="online", choices=["online", "offline"])
    # parser.add_argument('--llm_query_few_shot', type=str2bool, default=True, help="if True, ues few-shot to query LLM for answer")
    parser.add_argument('--wo_llm', type=str2bool, default=False, help="推理过程不使用LLM，只使用proxy，事后用LLM生成答案")

    parser.add_argument('--online_concurrency', type=int, default=64)
    parser.add_argument('--answer_temperature', type=float, default=0)
    parser.add_argument('--plan_temperature', type=float, default=0.3)
    parser.add_argument('--answer_top_p', type=float, default=1.)
    parser.add_argument('--plan_top_p', type=float, default=1.)

    ## params of file
    parser.add_argument("--debug_num", type=int, default=5)
    parser.add_argument('--datapath', type=str, default="./data")
    parser.add_argument('--dataname', type=str, default='2WikiMultiHopQA')  # 2WikiMultiHopQA NaturalQuestions
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument('--filter', type=str2bool, default=False)
    parser.add_argument('--filter_path', type=str, default="path to sampling_filter.json")
    parser.add_argument('--filter_key', type=str, default="")
    parser.add_argument('--focus_qid', type=str, default="")

    # params of multi_process
    parser.add_argument("--num_epoch", type=int, default=1)

    # eval answer (extract or compare)
    parser.add_argument('--only_eval_answer', type=str2bool, default=False)
    parser.add_argument('--extract_model_name', type=str, default="Qwen2-7B-Instruct")
    parser.add_argument('--batch_size', type=int, default=10000)

    # test 
    parser.add_argument('--test', type=str2bool, default=False)
    parser.add_argument('--test_file_name', type=str, default="test_1000")

    parser.add_argument('--backend', type=str, default="sglang", choices=["vllm", "sglang"])

    # planning cache
    parser.add_argument('--use_planning_cache', type=str2bool, default=True)
    parser.add_argument('--cache_dir', type=str, default="path to cache")

    ## params of single query
    parser.add_argument('--question', type=str, default="")

    args = parser.parse_args()  # 解析参数

    if args.filter_key == "None":
        args.filter_key = ""
        
    if args.focus_qid == "None":
        args.focus_qid = ""
    
    if args.wo_llm:
        assert args.use_planning_cache == True, "if wo_llm is True, use_planning_cache must be True"
        args.llm_server_url = None
    else:
        args.llm_server_url = args.llm_server_url.split(",")

    if args.test:
        args.temperature = 0
        assert args.model_type == "proxy", "model_type should be proxy when test is True"
        assert args.temperature == 0.0, "temperature should be 0.0 when test is True"
        args.max_tokens = 2048
        args.force_decision = False
        args.force_action = ""
        args.n_decision_sample = 1
        args.n_generate_sample = 1
        args.n_plan_sample = 1
        args.n_answer_sample = 1
        args.few_shot = False
        args.max_depth = 15
        args.data_file_name = f"{args.test_file_name}.jsonl"
        args.filter = False
        args.focus_qid = ""
        args.llm_server_type = "online"
    else:
        args.data_file_name = data_file_name[args.dataname]

    if args.dataname == "Musique":
        args.retrieve_server_url = args.musique_server_url
    

    if args.debug_num < 0 and args.llm_server_type == "offline":
        assert "Qwen2-72B-Instruct" in args.checkpoint_dir, "if llm_server_type is offline, checkpoint_dir must be Qwen2-72B-Instruct"


    args.timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if args.model_type == "proxy":
        if args.test:
            args.proxy_model_name = "/".join(args.checkpoint_dir.split("/")[-4:])
            args.output_dir = osp.join(args.output_dir, args.dataname, args.proxy_model_name, args.test_file_name, f"{args.timestamp}")
        else:
            args.proxy_model_name = osp.basename(args.checkpoint_dir)
            args.output_dir = osp.join(args.output_dir, args.dataname, args.proxy_model_name, args.timestamp)
    elif args.model_type == "gpt":
        args.output_dir = osp.join(args.output_dir, args.dataname, args.proxy_model_name, args.timestamp)

    # os.makedirs(args.output_dir, exist_ok=True)

    if args.seed == 0:
        seed_value = int(time.time()) & (2**32 - 1)  # 
        args.seed = seed_value

    # 启动log
    log_params(args, write_to_file=write_to_file)

    return args