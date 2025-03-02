import re

from typing import List

from prompt.decision import prompt_decision_making_few_shot, instruct_for_each_dataset, prompt_decision_making
from tree.node import BaseNode, State
from utils import few_shot_random_select

class MAKING_DECISION_ROLE():
    def __init__(self):
        self.legal_decision_action = [self.config['role']['MAKE_DECISION']['actions']['RETRIEVAL_ACTION'], self.config['role']['MAKE_DECISION']['actions']['NO_RETRIEVAL_ACTION'], self.config['role']['MAKE_DECISION']['actions']['PLANNING_ACTION']]

    def make_decision_role(self, nodes: List[BaseNode]):
        if len(nodes) == 0:
            return
        self.making_decision(nodes)

    def making_decision_prepare_input(self):  # 输入一定不会溢出
        nodes = self.make_decision_lst
        need_proxy_input = []
        if len(nodes) == 0:
            return need_proxy_input
        for node in nodes:
            if self.args.few_shot:
                info = {
                    "examples": few_shot_random_select(self.few_shot_examples, 'decision', num=self.args.few_num, dict_num=self.args.dict_few_num),
                    "question": self.tree.question,
                    "dataset_instructions": instruct_for_each_dataset.get(self.args.dataname, ""),
                }
                input_text = prompt_decision_making_few_shot.format_map(info)
            else:
                info = {
                    "question": self.tree.question,
                    "dataset_instructions": instruct_for_each_dataset.get(self.args.dataname, ""),
                }
                input_text = prompt_decision_making.format_map(info)

            if not self.args.force_decision:
                # input_template_text = self.organize_prompt(prompt=input_text)
                need_proxy_input.append({
                    "id": self.get_tree_node_tag(node),
                    "role": self.config['role']['MAKE_DECISION']['name'],
                    "need_generate": True,
                    "text": input_text,
                    "template_text": self.organize_message(prompt=input_text) if self.args.backend == "sglang" else self.organize_prompt(prompt=input_text)
                })
            else:
                input_template_text = None
                if self.args.force_action == "Planning":
                    outputs = [f"[Planning]"]
                elif self.args.force_action == "Retrieval":
                    outputs = [f"[Retrieval] {self.tree.question}"]
                elif self.args.force_action == "No Retrieval":
                    outputs = [f"[No Retrieval]"]
                else:
                    raise ValueError(f"Invalid force action: {self.args.force_action}")
                need_proxy_input.append({
                    "id": self.get_tree_node_tag(node),
                    "role": self.config['role']['MAKE_DECISION']['name'],
                    "need_generate": False,
                    "text": input_text,
                    "template_text": input_template_text,
                    "outputs": outputs
                })
            
        return need_proxy_input

    def making_decision_postprocess_output(self, outputs: dict):
        nodes = self.make_decision_lst
        if len(nodes) == 0:
            return 
        
        # parse output
        output_text = set() # do dudeplicate  local dudeplicate == global dudeplicate
        for node in nodes:  # len(nodes) must = 1
            tag = self.get_tree_node_tag(node)

            output = outputs[tag]
            for response in output['outputs']:
                if response in output_text:
                    continue
                output_text.add(response)
                new_state = self.parse_decision(output['text'], response)
                new_node = BaseNode(node_id=node.node_id + f".{len(node.children)}", parent=node, state=new_state, depth=node.depth+1, role=self.config['role']['MAKE_DECISION']['name'], plan=node.plan, documents=node.documents, query_history=node.query_history)
                node.add_child(new_node)

    def parse_decision(self, input_text:str, output_text: str) -> State:
        regex = r'\[(.*?)\](.*)'
        # state_list = []
        # for text in texts:
        match = re.search(regex, output_text, re.DOTALL)
        if match:
            action = match.group(1).strip()    # []中的文本
            content = match.group(2).strip()   # []后面的文本
            content = " ".join(content.split()[:self.args.max_query_length])

            state = State(input_text=input_text, output_text=output_text, action=action, action_input=content)
            if (action not in self.legal_decision_action) or (action==self.config['role']['MAKE_DECISION']['actions']['RETRIEVAL_ACTION'] and len(content)==0):
                state.is_terminal = True
                state.terminal_reason = "Invalid Action"

        else:
            state = State(input_text=input_text, output_text=output_text, is_terminal=True, terminal_reason="Invalid Format")

        # state_list.append(state)
        return state