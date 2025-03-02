import re

from typing import List

from prompt.decide_next_step import decide_next_step, decide_next_step_few_shot, decide_next_step_LLM, decide_next_step_LLM_few_shot
from tree.node import BaseNode, State
from utils import few_shot_random_select

import logging
# logger = logging.getLogger("my_custom_logger")
logger = logging.getLogger(__name__)

class DECIDE_NEXT_STEP():
    def __init__(self):
        self.legal_next_step = [self.config['role']['DECIDE_NEXT_STEP']['actions']['RETRIEVAL_ACTION'], self.config['role']['DECIDE_NEXT_STEP']['actions']['LLM_ACTION']]

    def decide_next_step_role(self, nodes: List[BaseNode]):
        if len(nodes) == 0:
            return
        self.decide_next_step(nodes)

    def decide_next_step_prepare_input(self):
        nodes = self.decide_next_step_lst
        need_proxy_input = []
        if len(nodes) == 0:
            return need_proxy_input
        
        for node in nodes:
            if self.args.few_shot:
                assert node.plan is not None, "Plan is None"
                if len(node.documents) > self.args.max_documents or node.depth > (self.args.max_depth - 4):  # 3 + 1 due to focusing on current step
                    llm_flag = True
                    logger.info(f"Too many documents {len(node.documents)} or too deep {node.depth}, expect to use LLM")
                    info = {
                        "examples": few_shot_random_select(self.few_shot_examples, 'decide_next_step', num=self.args.few_num, dict_num=self.args.dict_few_num, specified_key="LLM"),
                        'existing_documents': self.str_documents(node),
                        'planning': node.plan,
                        "question": self.tree.question,
                    }
                    input_text = decide_next_step_LLM_few_shot.format_map(info)
                else:
                    llm_flag = False
                    info = {
                        "examples": few_shot_random_select(self.few_shot_examples, 'decide_next_step', num=self.args.few_num, dict_num=self.args.dict_few_num, specified_key="no_existing_documents" if node.document_is_empty() else "has_existing_documents"),
                        'existing_documents': self.str_documents(node),
                        'planning': node.plan,
                        "question": self.tree.question,
                    }
                    input_text = decide_next_step_few_shot.format_map(info)
            else:
                assert node.plan is not None, "Plan is None"
                if len(node.documents) > self.args.max_documents or node.depth > (self.args.max_depth - 4):  # 3 + 1 due to focusing on current step
                    llm_flag = True
                    logger.info(f"Too many documents {len(node.documents)} or too deep {node.depth}, expect to use LLM")
                    info = {
                        'existing_documents': self.str_documents(node),
                        'planning': node.plan,
                        "question": self.tree.question,
                    }
                    input_text = decide_next_step_LLM.format_map(info)
                else:
                    llm_flag = False
                    info = {
                        'existing_documents': self.str_documents(node),
                        'planning': node.plan,
                        "question": self.tree.question,
                    }
                    input_text = decide_next_step.format_map(info)

            need_proxy_input.append({
                "id": self.get_tree_node_tag(node),
                "role": self.config['role']['DECIDE_NEXT_STEP']['name'],
                "need_generate": True,
                "text": input_text,
                "template_text": self.organize_message(prompt=input_text) if self.args.backend == "sglang" else self.organize_prompt(prompt=input_text),
                "LLM_flag": llm_flag,
            })

        return need_proxy_input

    def decide_next_step_postprocess_output(self, outputs: dict):
        nodes = self.decide_next_step_lst
        if len(nodes) == 0:
            return

        for node in nodes:
            tag = self.get_tree_node_tag(node)
            
            output = outputs[tag]
            output_text = set()
            for response in output['outputs']:
                if response in output_text:
                    continue
                output_text.add(response)
                new_state = self.parse_decide_next_step(output['text'], response, output['LLM_flag'])
                new_node = BaseNode(node_id=node.node_id + f".{len(node.children)}", parent=node, state=new_state, depth=node.depth+1, role=self.config['role']['DECIDE_NEXT_STEP']['name'], plan=node.plan, documents=node.documents, query_history=node.query_history)
                node.add_child(new_node)


    def decide_next_step(self, nodes: List[BaseNode]):
        input_lst, LLM_flag = [], []
        for node in nodes:
            if self.args.few_shot:
                assert node.plan is not None, "Plan is None"
                if len(node.documents) > self.args.max_documents or node.depth > (self.args.max_depth - 4):  # 3 + 1 due to focusing on current step
                    LLM_flag.append(True)
                    logger.info(f"Too many documents {len(node.documents)} or too deep {node.depth}, expect to use LLM")
                    info = {
                        "examples": few_shot_random_select(self.few_shot_examples, 'decide_next_step', num=self.args.few_num, dict_num=self.args.dict_few_num, specified_key="LLM"),
                        'existing_documents': self.str_documents(node),
                        'planning': node.plan,
                        "question": self.tree.question,
                    }
                    input_text = decide_next_step_LLM_few_shot.format_map(info)
                else:
                    LLM_flag.append(False)
                    info = {
                        "examples": few_shot_random_select(self.few_shot_examples, 'decide_next_step', num=self.args.few_num, dict_num=self.args.dict_few_num, specified_key="no_existing_documents" if node.document_is_empty() else "has_existing_documents"),
                        'existing_documents': self.str_documents(node),
                        'planning': node.plan,
                        "question": self.tree.question,
                    }
                    input_text = decide_next_step_few_shot.format_map(info)
            else:
                assert node.plan is not None, "Plan is None"
                if len(node.documents) > self.args.max_documents or node.depth > (self.args.max_depth - 4):  # 3 + 1 due to focusing on current step
                    LLM_flag.append(True)
                    logger.info(f"Too many documents {len(node.documents)} or too deep {node.depth}, expect to use LLM")
                    info = {
                        'existing_documents': self.str_documents(node),
                        'planning': node.plan,
                        "question": self.tree.question,
                    }
                    input_text = decide_next_step_LLM.format_map(info)
                else:
                    LLM_flag.append(False)
                    info = {
                        'existing_documents': self.str_documents(node),
                        'planning': node.plan,
                        "question": self.tree.question,
                    }
                    input_text = decide_next_step.format_map(info)
                
            input_lst.append(input_text)

        input_template_text_lst = self.organize_prompt(prompt=input_lst)
        outputs = self.get_proxy_outputs(input_template_text_lst, n=[self.n_generate_sample], role="decide_next_step")

        if outputs == "Timeout":
            for node in nodes:
                node.state.is_terminal = True
                node.state.terminal_reason = "Timeout"
        else:
            # parse output
            for i in range(len(nodes)):
                output_text = []
                for idx in range(len(outputs[i]["texts"])):
                    if outputs[i]["texts"][idx] in output_text:  # deplicate
                        continue
                    output_text.append(outputs[i]["texts"][idx])
                    new_state = self.parse_decide_next_step(input_lst[i], outputs[i]["texts"][idx], LLM_flag[i])
                    prior_prob = outputs[i]["prior_probs"][idx]
                    new_node = BaseNode(node_id=nodes[i].node_id + f".{len(nodes[i].children)}", parent=nodes[i], state=new_state, P=prior_prob, depth=nodes[i].depth+1, role=self.config['role']['DECIDE_NEXT_STEP']['name'], plan=nodes[i].plan, documents=nodes[i].documents)
                    nodes[i].add_child(new_node)
                    # logger.info(nodes[i].depth + 1)

    def parse_decide_next_step(self, input_text:str, output_text: str, LLM_flag: bool) -> State:
        # regex = r'Thought:\s*(.*?)\s*Action:\s*(.*)'
        regex = r'(?:Thought:\s*)?(.*?)\s*Action:\s*(.*)'
        regex_action = r'\[(.*?)\](.*)'

        match = re.search(regex, output_text, re.DOTALL)
        if match:
            thought_text = match.group(1).strip()
            action_text = match.group(2).strip()
            if len(thought_text.strip()) == 0:
                return State(input_text=input_text, output_text=output_text, is_terminal=True, terminal_reason="Invalid Format: Without Thought")
            match_action = re.search(regex_action, action_text, re.DOTALL)
            if match_action:
                action = match_action.group(1).strip()
                content = match_action.group(2).strip()
                content = " ".join(content.split()[:self.args.max_query_length])

                state = State(input_text=input_text, output_text=output_text, thought=thought_text, action=action, action_input=content)
                if LLM_flag:
                    if action != self.config['role']['DECIDE_NEXT_STEP']['actions']['LLM_ACTION']:
                        action = self.config['role']['DECIDE_NEXT_STEP']['actions']['LLM_ACTION']
                        state.terminal_reason = "May not proper [LLM]"
                    
                else:  # [Retrieval] 和 LLM都行
                    if (action not in self.legal_next_step) or (action==self.config['role']['DECIDE_NEXT_STEP']['actions']['RETRIEVAL_ACTION'] and len(content)==0):
                        state.is_terminal = True
                        state.terminal_reason = "Invalid Action"
            else:
                state = State(input_text=input_text, output_text=output_text, is_terminal=True, terminal_reason="Invalid Format")

        else:
            state = State(input_text=input_text, output_text=output_text, is_terminal=True, terminal_reason="Invalid Format")

        # state_list.append(state)
        return state



