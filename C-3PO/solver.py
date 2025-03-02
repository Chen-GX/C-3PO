import logging

logger = logging.getLogger(__name__)
import time
import json

from typing import List
from tqdm import tqdm
from pebble import ThreadPool

from proxy import load_proxy
from llm_server import LLM_SERVER
from search.search import Search
from retrieve.retriever import BasicRAG

def write_solutions_to_file(all_solutions, file_path):
    if file_path is None:
        return
    with open(file_path, 'a') as writer:
        for solution_item in all_solutions:
            writer.write(json.dumps(solution_item) + '\n')
            writer.flush()

class Solver():
    def __init__(self, args):
        self.args = args
        self.proxy = load_proxy(args)
        self.llm_server = LLM_SERVER(args) if self.args.llm_server_type == "online" else LLM_SERVER(args, llm=self.proxy.llm, sampling_params=self.proxy.sampling_params)
        self.n_generate_sample = args.n_generate_sample

        self.retrieve = BasicRAG(args)

        self.epoch_file_path = None
        self.final_file_path = None

    def solve(self, searches: List[Search]):

        search_length = len(searches)
        begin_time = time.time()
        all_save_tree_info = []
        node2documents = {}
        for i in tqdm(range(self.args.max_iter), desc="Step"):
            # if i in [3, 4, 6]:
            if i in [4, 6]:
                self.n_generate_sample = max(self.n_generate_sample - 1, 1)

            # prepare input
            all_proxy_input, all_llm_input = [], []
            for search in searches:
                proxy_input, llm_input = search.prepare_input(node2documents)
                all_proxy_input.extend(proxy_input)
                all_llm_input.extend(llm_input)
            
            # generate
            # all_proxy_input = self.proxy.generate(all_proxy_input, self.n_generate_sample)
            # all_proxy_input = {item["id"]: item for item in all_proxy_input}

            # all_llm_input = self.llm_server.generate(all_llm_input)
            # all_llm_input = {item["id"]: item for item in all_llm_input}

            with ThreadPool(max_workers=1) as pool:
                # llm_server 在线程中运行
                llm_future = pool.schedule(
                    self.llm_server.generate,
                    args=(all_llm_input,)
                )
                
                # proxy.generate (包含 Ray 调用)在主线程运行
                proxy_result = self.proxy.generate(all_proxy_input, self.n_generate_sample)
                # 获取 llm_server 结果
                llm_result = llm_future.result()
            all_proxy_input = {item["id"]: item for item in proxy_result}
            all_llm_input = {item["id"]: item for item in llm_result}

            # postprocess output
            finish_searches = []
            for search in searches:
                search.postprocess_output(all_proxy_input, all_llm_input)
                if search.search_finish():
                    finish_searches.append(search)
            
            for search in finish_searches:
                searches.remove(search)
            # [v for v in all_proxy_input.values() if '[Retrieval]' in v['outputs'][0]] # [v for v in all_proxy_input.values() if '[Planning]' in v['outputs'][0]]
            # do retrieve
            node2documents = self.batch_retrieve(searches)
            
            # save_solution
            all_extract_answer_input = []
            if len(finish_searches) > 0:
                for search in finish_searches:
                    extract_answer_input = search.save_solution_prepare_input()
                    all_extract_answer_input.extend(extract_answer_input)
                
                all_extract_answer_input = self.llm_server.generate(all_extract_answer_input)
                all_extract_answer_input = {item["id"]: item for item in all_extract_answer_input}

                save_tree_info_list = []
                for search in finish_searches:
                    save_tree_info = search.save_solution_postprocess_output(all_extract_answer_input)
                    save_tree_info_list.append(save_tree_info)
                
                write_solutions_to_file(save_tree_info_list, self.epoch_file_path)
                all_save_tree_info.extend(save_tree_info_list)
    
        logger.info(f"Epoch finished. Save all solutions")
        write_solutions_to_file(all_save_tree_info, self.final_file_path)
        logger.info(f"Epoch finished. Time: {time.time() - begin_time}. Len: {search_length}")
        return all_save_tree_info
    
    def batch_retrieve(self, searches, batch_size=4096):
        retrieve_input = {"node_id": [], "queries": [], "retrieve_times": []}
        for search in searches:
            node_id, queries, retrieve_times = search.do_retrieve_prepare_input()
            retrieve_input["node_id"].extend(node_id)
            retrieve_input["queries"].extend(queries)
            retrieve_input["retrieve_times"].extend(retrieve_times)

        node_id_list, query_list, documents_list = [], [], []
        if len(retrieve_input["node_id"]) > 0:
            # 批量处理
            total_queries = len(retrieve_input["queries"])
            for i in range(0, total_queries, batch_size):  # TODO: 多线程 多url
                # 获取当前批次的查询和检索时间
                batch_node_id = retrieve_input["node_id"][i:i + batch_size]
                batch_queries = retrieve_input["queries"][i:i + batch_size]
                batch_retrieve_times = retrieve_input["retrieve_times"][i:i + batch_size]
                
                # 执行检索
                batch_documents = self.retrieve.retrieve(batch_queries, batch_retrieve_times)
                
                # 将当前批次的结果添加到总的文档列表中
                node_id_list.extend(batch_node_id)
                query_list.extend(batch_queries)
                documents_list.extend(batch_documents)
            
            # 处理searches
            assert len(node_id_list) == len(documents_list), f"{len(node_id_list)} != {len(documents_list)}"
            node2documents = {node_id: {"query": query, "documents": documents} for node_id, query, documents in zip(node_id_list, query_list, documents_list)}
        else:
            node2documents = {}

        return node2documents

