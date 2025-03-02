import re

from typing import List

from tree.node import BaseNode, State

from prompt.filter import FILTER_PROMPT, FILTER_PROMPT_FEW_SHOT, FILTER_PROMPT_DIRECT_RETRIEVE, FILTER_PROMPT_DIRECT_RETRIEVE_FEW_SHOT
from utils import few_shot_random_select

class RETRIEVE_AND_FILTER_ROLE():
    def __init__(self):
        pass

    def do_retrieve_prepare_input(self):
        nodes = self.retrieve_and_filter_lst
        if len(nodes) == 0:
            return [], [], []
        
        node_id, queries, retrieve_times = [], [], []
        for node in nodes:
            assert node.state.action in [self.config['role']['MAKE_DECISION']['actions']['RETRIEVAL_ACTION'], self.config['role']['DECIDE_NEXT_STEP']['actions']['RETRIEVAL_ACTION']], f"Invalid action: {node.state.action}"
            node_id.append(self.get_tree_node_tag(node))
            queries.append(node.state.action_input)
            retrieve_times.append(node.query_history.get(node.state.action_input, 0))
        # documents = self.retrieve.retrieve(queries, retrieve_times)
        return node_id, queries, retrieve_times

    
    def do_filter_prepare_input(self, node2documents):
        nodes = self.retrieve_and_filter_lst
        if len(nodes) == 0:
            return []
        
        documents = []
        for node in nodes:
            assert node.state.action in [self.config['role']['MAKE_DECISION']['actions']['RETRIEVAL_ACTION'], self.config['role']['DECIDE_NEXT_STEP']['actions']['RETRIEVAL_ACTION']], f"Invalid action: {node.state.action}"
            node_id = self.get_tree_node_tag(node)
            assert node_id in node2documents, f"Invalid node_id: {node_id}"
            assert node.state.action_input == node2documents[node_id]['query'], f"Invalid query: {node.state.action_input} != {node2documents[node_id]['query']}"
            documents.append(node2documents[node_id]['documents'])

        return self.filter_prepare_input(nodes, documents)
    

    def filter_prepare_input(self, nodes: List[BaseNode], documents: List[dict]):
        need_proxy_input = []
        for node, document in zip(nodes, documents):
            if self.args.few_shot:
                if node.role == self.config['role']['MAKE_DECISION']['name']:
                    infos = {
                        "examples": few_shot_random_select(self.few_shot_examples, 'filter', num=self.args.few_num, dict_num=self.args.dict_few_num, specified_key="retrieve"),
                        "question": self.tree.question,
                        "documents": self.str_documents(document),
                    }
                    input_text = FILTER_PROMPT_DIRECT_RETRIEVE_FEW_SHOT.format_map(infos)
                elif node.role == self.config['role']['DECIDE_NEXT_STEP']['name']:
                    infos = {
                        "examples": few_shot_random_select(self.few_shot_examples, 'filter', num=self.args.few_num, dict_num=self.args.dict_few_num, specified_key="planning"),
                        "objective": node.state.thought if node.state.thought is not None else '',
                        "question": self.tree.question,
                        "documents": self.str_documents(document),
                    }
                    input_text = FILTER_PROMPT_FEW_SHOT.format_map(infos)
                else:
                    raise ValueError(f"Invalid role in filter action: {node.role}")
            else:
                if node.role == self.config['role']['MAKE_DECISION']['name']:
                    infos = {
                        "question": self.tree.question,
                        "documents": self.str_documents(document),
                    }
                    input_text = FILTER_PROMPT_DIRECT_RETRIEVE.format_map(infos)
                elif node.role == self.config['role']['DECIDE_NEXT_STEP']['name']:
                    infos = {
                        "objective": node.state.thought if node.state.thought is not None else '',
                        "question": self.tree.question,
                        "documents": self.str_documents(document),
                    }
                    input_text = FILTER_PROMPT.format_map(infos)
                else:
                    raise ValueError(f"Invalid role in filter action: {node.role}")
            
            # valid_index.append()
            # input_lst.append(input_text)
            need_proxy_input.append({
                "id": self.get_tree_node_tag(node),
                "role": self.config['role']['RETRIEVE_AND_FILTER']['name'],
                "need_generate": True,
                "text": input_text,
                "template_text": self.organize_message(prompt=input_text) if self.args.backend == "sglang" else self.organize_prompt(prompt=input_text),
                "valid_index": len(document),
                "document": document,
            })
            
        return need_proxy_input

    def filter_postprocess_output(self, outputs: dict):
        nodes = self.retrieve_and_filter_lst
        if len(nodes) == 0:
            return
        
        for node in nodes:
            tag = self.get_tree_node_tag(node)
            node.query_history[node.state.action_input] = node.query_history.get(node.state.action_input, 0) + 1
            output = outputs[tag]
            output_text = set()
            for response in output['outputs']:
                if response in output_text:
                    continue
                output_text.add(response)
                new_state = self.parse_filter(output['text'], response, output['valid_index'], node)
                if not new_state.is_terminal:
                    new_state.retrieve_query = node.state.action_input
                    new_state.retrieved_all_documents = output['document']
                    if new_state.action_input == []:
                        select_documents = [{"title": node.state.action_input, "text": "None relevant information about " + node.state.action_input}]
                    else:
                        select_documents = [output['document'][select_idx - 1] for select_idx in new_state.action_input]  # select_idx need minus 1 as index
                    
                new_node = BaseNode(node_id=node.node_id + f".{len(node.children)}", parent=node, state=new_state, depth=node.depth+1, role=self.config['role']['RETRIEVE_AND_FILTER']['name'], plan=node.plan, documents=node.documents, query_history=node.query_history)

                if not new_state.is_terminal:
                    new_node.extend_documents(select_documents)
                    # 检查当前query是否消除了之前None relevant的document
                    if new_state.action_input != [] and new_node.query_history[new_state.retrieve_query] > 1:
                        # 当前query已经被使用过，并且有新的document被选中
                        new_node.remove_noinfo_documents("None relevant information about " + new_state.retrieve_query)
                        

                node.add_child(new_node)


    def do_retrieve_and_filter(self, nodes: List[BaseNode]):
        
        # for node, document in zip(nodes, documents):
        #     node.state.observation = document  # List[{"text": xxx, "title": xxx, ...}]
        self.do_filter(nodes, documents)
        

    def do_filter(self, nodes: List[BaseNode], documents: List[dict]):
        input_lst, valid_index = [], []
        for node, document in zip(nodes, documents):
            if self.args.few_shot:
                if node.role == self.config['role']['MAKE_DECISION']['name']:
                    infos = {
                        "examples": few_shot_random_select(self.few_shot_examples, 'filter', num=self.args.few_num, dict_num=self.args.dict_few_num, specified_key="retrieve"),
                        "question": self.tree.question,
                        "documents": self.str_documents(document),
                    }
                    input_text = FILTER_PROMPT_DIRECT_RETRIEVE_FEW_SHOT.format_map(infos)
                elif node.role == self.config['role']['DECIDE_NEXT_STEP']['name']:
                    infos = {
                        "examples": few_shot_random_select(self.few_shot_examples, 'filter', num=self.args.few_num, dict_num=self.args.dict_few_num, specified_key="planning"),
                        "objective": node.state.thought if node.state.thought is not None else '',
                        "question": self.tree.question,
                        "documents": self.str_documents(document),
                    }
                    input_text = FILTER_PROMPT_FEW_SHOT.format_map(infos)
                else:
                    raise ValueError(f"Invalid role in filter action: {node.role}")
            else:
                if node.role == self.config['role']['MAKE_DECISION']['name']:
                    infos = {
                        "question": self.tree.question,
                        "documents": self.str_documents(document),
                    }
                    input_text = FILTER_PROMPT_DIRECT_RETRIEVE.format_map(infos)
                elif node.role == self.config['role']['DECIDE_NEXT_STEP']['name']:
                    infos = {
                        "objective": node.state.thought if node.state.thought is not None else '',
                        "question": self.tree.question,
                        "documents": self.str_documents(document),
                    }
                    input_text = FILTER_PROMPT.format_map(infos)
                else:
                    raise ValueError(f"Invalid role in filter action: {node.role}")
            
            valid_index.append(len(document))
            input_lst.append(input_text)
            
        input_template_text_lst = self.organize_prompt(prompt=input_lst)
        # query proxy
        outputs = self.get_proxy_outputs(input_template_text_lst, n=[self.n_generate_sample], role="filter")

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
                    # outputs[i]["texts"][idx] = """Thought: Document 1: Relevant. It mentions that "The Sea Wolf (1920 film)" was directed by George Melford and is an American film.\nDocument 2: Partially relevant. It repeats the information from Document 1 about "The Sea Wolf (1920 film)" being an American film directed by George Melford.\nDocument 3: Not relevant. It discusses the plot of a different version of "The Sea Wolf" film, not the 1920 version.\nDocument 4: Not relevant. It's about a 1913 version of "The Sea Wolf" film.\nDocument 5: Partially relevant. It provides more plot details about the 1920 film but doesn't add new information about the director or nationality.\nDocument 6: Not relevant. It's about the 1913 version of "The Sea Wolf" film.\nDocument 7: Not relevant. It's about a 1975 Italian adaptation of "The Sea Wolf" novel.\nDocument 8: Not relevant. It's about a 1941 version of "The Sea Wolf" film.\nDocument 9: Not relevant. It provides plot details for the 1920 film but doesn't add information about the director or nationality.\nDocument 10: Not relevant. It's about the 1941 version of "The Sea Wolf" film.\nAfter analysis, the most relevant documents are 1 and 2, as they provide information about the director and nationality of "The Sea Wolf (1920 film)". However, none of the documents contain information about "The Nines" or its director.\n\nAction: [1, 2]"""
                    new_state = self.parse_filter(input_lst[i], outputs[i]["texts"][idx], valid_index[i], nodes[i])
                    if not new_state.is_terminal:
                        new_state.retrieve_query = nodes[i].state.action_input
                        new_state.retrieved_all_documents = documents[i]
                        if new_state.action_input == []:
                            select_documents = [{"title": nodes[i].state.action_input, "text": "None relevant information about " + nodes[i].state.action_input}]
                        else:
                            select_documents = [documents[i][select_idx - 1] for select_idx in new_state.action_input]  # select_idx need minus 1 as index
            
                    prior_prob = outputs[i]["prior_probs"][idx]
                    new_node = BaseNode(node_id=nodes[i].node_id + f".{len(nodes[i].children)}", parent=nodes[i], state=new_state, P=prior_prob, depth=nodes[i].depth+1, role=self.config['role']['RETRIEVE_AND_FILTER']['name'], plan=nodes[i].plan, documents=nodes[i].documents)
        
                    if not new_state.is_terminal:
                        new_node.extend_documents(select_documents)

                    nodes[i].add_child(new_node)

    
    def parse_filter(self, input_text:str, output_text: str, valid_index: int, node: BaseNode) -> State:
        # regex = r'Thought:\s*(.*?)\s*Action:\s*(.*)'
        regex = r'(?:Thought:\s*)?(.*?)\s*Action:\s*(.*)'
        regex_action = r'\[(.*?)\](.*)'

        match = re.search(regex, output_text, re.DOTALL)
        if match:
            thought_text = match.group(1).strip()
            action_text = match.group(2).strip()
            if len(thought_text.strip()) == 0:
                return State(input_text=input_text, output_text=output_text, thought=thought_text, is_terminal=True, terminal_reason="Invalid Format: Without Thought")
                
            match_action = re.search(regex_action, action_text, re.DOTALL)
            if match_action:
                content = match_action.group(1).strip().split(',')  # Action: [1, 2]
                if len(content) == 0:
                    return State(input_text=input_text, output_text=output_text, thought=thought_text, is_terminal=True, terminal_reason="Invalid Action")

                select_idx = []
                for idx in content:
                    try:
                        if idx == '':
                            continue
                        idx = int(idx.strip())
                    except:
                        return State(input_text=input_text, output_text=output_text, thought=thought_text, is_terminal=True, terminal_reason="Invalid Document Idx")
                    
                    if idx in range(1, valid_index + 1):
                        select_idx.append(idx)
                    else:
                        if not self.args.test:
                            # 数据采集阶段，严格一点，对于幻觉出来的idx直接termnial
                            return State(input_text=input_text, output_text=output_text, thought=thought_text, is_terminal=True, terminal_reason="Hallucinate Document Idx")
                
                select_idx = list(dict.fromkeys(select_idx))

                if node.role == self.config['role']['MAKE_DECISION']['name']:
                    state = State(input_text=input_text, output_text=output_text, action=self.config['role']['RETRIEVE_AND_FILTER']['actions']['DIRECT_RETRIEVE_AND_FILTER_ACTION'], thought=thought_text, action_input=select_idx)
                elif node.role == self.config['role']['DECIDE_NEXT_STEP']['name']:
                    state = State(input_text=input_text, output_text=output_text, action=self.config['role']['RETRIEVE_AND_FILTER']['actions']['RETRIEVE_AND_FILTER_ACTION'], thought=thought_text, action_input=select_idx)
                else:
                    raise ValueError(f"Invalid role in filter action: {node.role}")
            
            else:
                return State(input_text=input_text, output_text=output_text, thought=thought_text, is_terminal=True, terminal_reason="Invalid Action")

        else:
            state = State(input_text=input_text, output_text=output_text, is_terminal=True, terminal_reason="Invalid Format")
        
        return state



