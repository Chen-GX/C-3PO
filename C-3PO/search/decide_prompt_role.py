import asyncio
import aiohttp

from typing import List

from prompt.decide_prompt_identity import NO_DOCUMENT_PROMPT, DOCUMENT_PROMPT, instruct_for_each_dataset
from prompt.decide_prompt_identity_v2 import DOCUMENT_PROMPT_FEW_SHOT

from tree.node import BaseNode, State

from utils import few_shot_random_select


class DECIDE_PROMPT_ROLE():
    def __init__(self):
        super().__init__()

    def decide_prompt_prepare_input(self):
        nodes = self.decide_prompt_lst
        if len(nodes) == 0:
            return []
        
        if self.args.decide_prompt == "identity":
            return self.identity_prompt(nodes)
        else:
            raise ValueError(f"Invalid decide_prompt method: {self.args.decide_prompt}")


    def identity_prompt(self, nodes: List[BaseNode]):
        need_proxy_input = []
        # input_text_lst, message_lst, model_url_lst = [], [], []
        for i, node in enumerate(nodes):
            if node.document_is_empty():
                infos = {
                    "question": self.tree.question,
                    "dataset_instructions": instruct_for_each_dataset.get(self.args.dataname, "")
                }
                input_text = NO_DOCUMENT_PROMPT.format_map(infos)
            else:
                if self.args.llm_query_few_shot:
                    infos = {
                        "documents": self.str_documents(node),
                        "dataset_instructions": instruct_for_each_dataset.get(self.args.dataname, ""),
                        "question": self.tree.question,
                        "examples": few_shot_random_select(self.few_shot_examples, 'answer', num=self.args.few_num, dict_num=1),
                    }
                    input_text = DOCUMENT_PROMPT_FEW_SHOT.format_map(infos)
                else:
                    infos = {
                        "documents": self.str_documents(node),
                        "dataset_instructions": instruct_for_each_dataset.get(self.args.dataname, ""),
                        "question": self.tree.question,
                    }
                    input_text = DOCUMENT_PROMPT.format_map(infos)
            
            message =[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text},
            ]

            need_proxy_input.append({
                "id": self.get_tree_node_tag(node),
                "role": self.config['role']['DECIDE_PROMPT']['name'],
                "need_generate": False,
                "text": input_text,
                "message": message,
                "generate_config": {
                    "n": self.n_answer_sample,
                    "temperature": self.args.answer_temperature,
                    "top_p": self.args.answer_top_p,
                }
            })
        return need_proxy_input

    def decide_prompt_postprocess_output(self, outputs: dict):
        nodes = self.decide_prompt_lst
        if len(nodes) == 0:
            return
        
        if self.args.decide_prompt in ["identity", "identity_few_shot"]:
            self.identity_prompt_postprocess_output(outputs)
        else:
            raise ValueError(f"Invalid decide_prompt method: {self.args.decide_prompt}")
        

    def identity_prompt_postprocess_output(self, outputs: dict):
        nodes = self.decide_prompt_lst
        for node in nodes:
            tag = self.get_tree_node_tag(node)
            output = outputs[tag]
            new_state = State(output_text=output['text'], action=self.config['role']['DECIDE_PROMPT']['actions']['DECIDE_PROMPT_ACTION'], action_input=output['message'], thought=output['generate_config']) # 
            new_node = BaseNode(node_id=node.node_id + f".{len(node.children)}", parent=node, state=new_state, depth=node.depth+1, role=self.config['role']['DECIDE_PROMPT']['name'], plan=node.plan, documents=node.documents, query_history=node.query_history)
            node.add_child(new_node)

