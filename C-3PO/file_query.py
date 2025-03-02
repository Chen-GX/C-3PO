import os, sys
os.chdir(sys.path[0])
import os.path as osp
import random
import yaml
import time

from tqdm import tqdm
from transformers import AutoTokenizer

from types import SimpleNamespace

from solver import Solver
from search.search import Search
from utils import print_tree, load_jsonl


if __name__=="__main__":
    try:
        # load yaml  CUDA_VISIBLE_DEVICES=0
        # with open("./config/math500_server.yaml", "r") as f:
        with open("./config/freshqa_server.yaml", "r") as f:
        # with open("./config/multihoprag_server.yaml", "r") as f:
            args = SimpleNamespace(**yaml.load(f, Loader=yaml.FullLoader))

        if "dashscope" in args.llm_server_url:
            args.online_concurrency = 4

        args.llm_server_url = args.llm_server_url.split(',')
        args.timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        solver = Solver(args)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
        ## refine save path
        epoch = 0
        if args.force_decision:
            args.output_dir = osp.join(args.output_dir, f"{args.model_base_name}_{args.max_depth}", args.force_action, f"{args.llm_name}", args.timestamp)
        else:
            args.output_dir = osp.join(args.output_dir, f"{args.model_base_name}_{args.max_depth}", f"{args.llm_name}_{args.retriever_type}", args.timestamp)
        file_path = osp.join(args.output_dir, "collectted_solutions")
        os.makedirs(file_path, exist_ok=True)
        epoch_file_path = osp.join(file_path, f'collectted_solutions_{epoch}.jsonl')
        solver.epoch_file_path = epoch_file_path
        solver.final_file_path = osp.join(file_path, f"{epoch}_final_result.jsonl")

        # check cache 
        if args.use_planning_cache:
            solver.llm_server.check_cache(data)

        # load file
        data = load_jsonl(args.data_path)


        trees = [Search(args=args, data_item=item, tree_tag=str(idx), tokenizer=tokenizer, server=False) for idx, item in enumerate(data)]
        all_save_tree_info = solver.solve(trees)

        solver.retrieve.retriever.save_cache()
        # print_tree(all_save_tree_info)

        # 把args保存到文件
        with open(osp.join(args.output_dir, "args.yaml"), "w") as f:
            yaml.dump(vars(args), f)


        if args.model_type == "proxy" and args.backend == "sglang":
            from sglang.utils import terminate_process
            terminate_process(solver.proxy.server_process)
             
    except Exception as e:
        print(e)
        if args.model_type == "proxy" and args.backend == "sglang":
            from sglang.utils import terminate_process
            terminate_process(solver.proxy.server_process)
        raise e