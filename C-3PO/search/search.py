import re
import sys
import time
import yaml
import random
import importlib
import asyncio

from typing import List, Dict, Union

from copy import deepcopy

from tree.tree import Tree
from tree.node import BaseNode, State

from search.make_decision_role import MAKING_DECISION_ROLE
from search.LLM_planning_role import LLM_PLANNING_ROLE
from search.decide_next_step_role import DECIDE_NEXT_STEP
from search.retrieve_filter_role import RETRIEVE_AND_FILTER_ROLE
from search.decide_prompt_role import DECIDE_PROMPT_ROLE
from search.LLM_query_role import LLM_QUERY_ROLE
from search.extract_answer_role import EXTRACT_ANSWER_ROLE
from search.evaluation_role import EVALUATION_ROLE

# from retrieve.retriever import BasicRAG
# from llm_server import LLM_SERVER

from utils import load_few_shot 

import logging

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS_PER_REQUEST = 600
TIMEOUT_MESSAGE_PER_REQUEST = f"Execution of vllm decoding has timed out for exceeding {TIMEOUT_SECONDS_PER_REQUEST} seconds."


class Search(MAKING_DECISION_ROLE, LLM_PLANNING_ROLE, DECIDE_NEXT_STEP, RETRIEVE_AND_FILTER_ROLE, DECIDE_PROMPT_ROLE, LLM_QUERY_ROLE, EXTRACT_ANSWER_ROLE, EVALUATION_ROLE):

    def __init__(self, args, data_item, tree_tag: str="0", tokenizer=None, server=False):
        # load config
        with open(args.search_config, 'r') as f:
            self.config = yaml.safe_load(f)

        MAKING_DECISION_ROLE.__init__(self)
        # LLM_PLANNING_ROLE.__init__(self)
        DECIDE_NEXT_STEP.__init__(self)
        # RETRIEVE_AND_FILTER_ROLE.__init__(self)
        # DECIDE_PROMPT_ROLE.__init__(self)

        super().__init__()
        self.tree_tag = tree_tag
        self.args = deepcopy(args)
        self.server = server
        if not server:
            if self.args.dataname == "Musique":
                max_depth = int(data_item['origin_id'][0])
                self.args.max_depth = max(2 * max_depth + 5, self.args.max_depth)
        
        self.n_decision_sample = args.n_decision_sample
        self.n_generate_sample = args.n_generate_sample
        self.n_plan_sample = args.n_plan_sample
        self.n_answer_sample = args.n_answer_sample
        self.tree = Tree(self.args, data_item, server=server)
        if self.args.model_type == "proxy":
            # self.tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
            self.tokenizer = tokenizer

        # self.retrieve = BasicRAG(args)
        # self.retrieve = retrieve
        # self.llm = LLM_SERVER(args)  # 一定是自己部署的模型

        self.few_shot_examples = load_few_shot(args, config=self.config)

        self.init_search()

    def organize_prompt(self, prompt: Union[List, str], system_prompt: str="You are a helpful assistant.") -> Union[str, List[str]]:
        if self.args.model_type == "proxy":
            if isinstance(prompt, str):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            elif isinstance(prompt, List):
                text = []
                for p in prompt:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": p}
                    ]
                    text.append(self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    ))
        elif self.args.model_type == "gpt":
            text = prompt

        return text
    
    def organize_message(self, prompt: str, system_prompt: str="You are a helpful assistant.") -> Union[str, List[str]]:
        return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

    def str_documents(self, node: Union[BaseNode, List]) -> str:
        if isinstance(node, BaseNode):
            return "\n\n".join([f"Document {idx+1} (Title: {item['title']}): {item['text']}" for idx, item in enumerate(node.documents)])
        elif isinstance(node, List):
            return "\n\n".join([f"Document {idx+1} (Title: {item['title']}): {item['text']}" for idx, item in enumerate(node)])
        else:
            raise ValueError("Invalid Type")

    def init_search(self):
        self.make_decision_lst, self.planning_lst, self.decide_next_step_lst, self.retrieve_and_filter_lst, self.decide_prompt_lst, self.llm_query_lst = [self.tree.root], [], [], [], [], []

    def get_tree_node_tag(self, node):
        return f"{self.tree_tag}@{node.node_id}"
    
    def get_extract_answer_tag(self, num):
        return f"{self.tree_tag}@{num}"

    def prepare_input(self, node2documents):
        need_proxy_input, need_llm_input = [], []
        # prepare input for proxy and llm
        make_decision_proxy_input_list = self.making_decision_prepare_input()  # sent to proxy
        need_proxy_input.extend(make_decision_proxy_input_list)

        llm_planning_llm_input_list = self.llm_planning_prepare_input()
        need_llm_input.extend(llm_planning_llm_input_list)

        decide_next_step_proxy_input_list = self.decide_next_step_prepare_input()
        need_proxy_input.extend(decide_next_step_proxy_input_list)

        retrieve_and_filter_proxy_input_list = self.do_filter_prepare_input(node2documents)
        need_proxy_input.extend(retrieve_and_filter_proxy_input_list)

        decide_prompt_proxy_input_list = self.decide_prompt_prepare_input()
        need_proxy_input.extend(decide_prompt_proxy_input_list)
        
        llm_query_llm_input_list = self.llm_query_prepare_input()
        need_llm_input.extend(llm_query_llm_input_list)

        return need_proxy_input, need_llm_input

    def postprocess_output(self, proxy_output, llm_output):
        # parse text and finish current step
        self.making_decision_postprocess_output(proxy_output)

        self.llm_planning_postprocess_output(llm_output)

        self.decide_next_step_postprocess_output(proxy_output)

        self.filter_postprocess_output(proxy_output)

        self.decide_prompt_postprocess_output(proxy_output)

        self.llm_query_postprocess_output(llm_output)
        
        self.update_next_round_node_list()
        
    def save_solution_prepare_input(self):
        # extract save the tree
        self.save_tree_info = self.tree.save_tree()
        if self.args.answer_eval == "extract":
            return self.extract_answer_prepare_input()
        elif self.args.answer_eval == "compare":
            return self.evluation_prepare_input()
        else:
            return []
    
    def save_solution_postprocess_output(self, llm_output):
        if self.args.answer_eval == "extract":
            self.extract_answer_postprocess_output(llm_output)
        elif self.args.answer_eval == "compare":
            self.evluation_postprocess_output(llm_output)
        return self.save_tree_info

    def get_cur_node_lst(self):
        return self.make_decision_lst + self.planning_lst + self.decide_next_step_lst + self.retrieve_and_filter_lst + self.decide_prompt_lst + self.llm_query_lst

    def search_finish(self):
        return len(self.get_cur_node_lst()) == 0

    def update_next_round_node_list(self):
        cur_node_lst = self.get_cur_node_lst()
        make_decision_lst, planning_lst, decide_next_step_lst, retrieve_and_filter_lst, decide_prompt_lst, llm_query_lst = [], [], [], [], [], []
        for cur_node in cur_node_lst:
            for child in cur_node.children:
                if child.depth > self.args.max_depth:
                    child.state.is_terminal = True
                    child.state.terminal_reason = "Max Depth"
            
                if not child.state.is_terminal:
                    # decision making
                    if child.role == self.config['role']['MAKE_DECISION']['name']:
                        if child.state.action == self.config['role']['MAKE_DECISION']['actions']['PLANNING_ACTION']:
                            planning_lst.append(child)
                        elif child.state.action == self.config['role']['MAKE_DECISION']['actions']['RETRIEVAL_ACTION']:
                            retrieve_and_filter_lst.append(child)
                        elif child.state.action == self.config['role']['MAKE_DECISION']['actions']['NO_RETRIEVAL_ACTION']:
                            decide_prompt_lst.append(child)
                        else:
                            # raise ValueError(f"Invalid Action in {child.role}: {child.state.action}")
                            child.state.is_terminal = True
                            child.state.terminal_reason = "Invalid Action"
                    # llm planning
                    elif child.role == self.config['role']['LLM_PLANNING']['name']:
                        if child.state.action == self.config['role']['LLM_PLANNING']['actions']['LLM_PLANNING_ACTION']:
                            decide_next_step_lst.append(child)
                        else:
                            # raise ValueError(f"Invalid Action in {child.role}: {child.state.action}")
                            child.state.is_terminal = True
                            child.state.terminal_reason = "Invalid Action"
                    # decide_next_step
                    elif child.role == self.config['role']['DECIDE_NEXT_STEP']['name']:
                        if child.state.action == self.config['role']['DECIDE_NEXT_STEP']['actions']['RETRIEVAL_ACTION']:
                            retrieve_and_filter_lst.append(child)
                        elif child.state.action == self.config['role']['DECIDE_NEXT_STEP']['actions']['LLM_ACTION']:
                            decide_prompt_lst.append(child)
                        else:
                            # raise ValueError(f"Invalid Action in {child.role}: {child.state.action}")
                            child.state.is_terminal = True
                            child.state.terminal_reason = "Invalid Action"
                    # filter_role
                    elif child.role == self.config['role']['RETRIEVE_AND_FILTER']['name']:
                        if child.state.action == self.config['role']['RETRIEVE_AND_FILTER']['actions']['RETRIEVE_AND_FILTER_ACTION']:
                            decide_next_step_lst.append(child)
                        elif child.state.action == self.config['role']['RETRIEVE_AND_FILTER']['actions']['DIRECT_RETRIEVE_AND_FILTER_ACTION']:
                            decide_prompt_lst.append(child)
                        else:
                            # raise ValueError(f"Invalid Action in {child.role}: {child.state.action}")
                            child.state.is_terminal = True
                            child.state.terminal_reason = "Invalid Action"
                    # decide_prompt
                    elif child.role == self.config['role']['DECIDE_PROMPT']['name']:
                        if child.state.action == self.config['role']['DECIDE_PROMPT']['actions']['DECIDE_PROMPT_ACTION']:
                            llm_query_lst.append(child)
                        else:
                            # raise ValueError(f"Invalid Action in {child.role}: {child.state.action}")
                            child.state.is_terminal = True
                            child.state.terminal_reason = "Invalid Action"
                    else:
                        child.state.is_terminal = True
                        child.state.terminal_reason = "Invalid Action"
        
        self.make_decision_lst, self.planning_lst, self.decide_next_step_lst, self.retrieve_and_filter_lst, self.decide_prompt_lst, self.llm_query_lst = make_decision_lst, planning_lst, decide_next_step_lst, retrieve_and_filter_lst, decide_prompt_lst, llm_query_lst

