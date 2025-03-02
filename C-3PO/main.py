import os, sys
os.chdir(sys.path[0])
import os.path as osp
import random
import logging

logger = logging.getLogger(__name__)

from tqdm import tqdm
from transformers import AutoTokenizer

from solver import Solver
from search.search import Search
from arguments import get_args, set_seed
from utils import load_data, create_batches

if __name__=="__main__":
    try:
        args = get_args(write_to_file=True)
        set_seed(args.seed)
        data = load_data(args)
        logger.info(f"{args.dataname} data loaded, length: {len(data)}")

        file_path = osp.join(args.output_dir, "collectted_solutions")
        os.makedirs(file_path, exist_ok=True)

        solver = Solver(args)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)

        # check cache
        if args.use_planning_cache:
            solver.llm_server.check_cache(data)

        for epoch in range(args.num_epoch):
            if not args.test:
                random.shuffle(data)
            epoch_file_path = osp.join(file_path, f'collectted_solutions_{epoch}.jsonl')
            solver.epoch_file_path = epoch_file_path
            solver.final_file_path = osp.join(file_path, f"{epoch}_final_result.jsonl")
            logger.info(f"********** EPOCH {epoch} ***********")
            batch_num = 0
            for batch_data in tqdm(create_batches(data, args.batch_size), desc="batch data"):
                batch_num += 1
                logger.info(f"begin {batch_num}, {len(batch_data)}")
                sys.stdout.flush()
                trees = [Search(args=args, data_item=item, tree_tag=str(idx), tokenizer=tokenizer) for idx, item in enumerate(batch_data)]
                solver.solve(trees)
                
        if args.model_type == "proxy" and args.backend == "sglang":
            from sglang.utils import terminate_process
            terminate_process(solver.proxy.server_process)
    except Exception as e:
        logger.exception(e)
        if args.model_type == "proxy" and args.backend == "sglang":
            from sglang.utils import terminate_process
            terminate_process(solver.proxy.server_process)
        raise e