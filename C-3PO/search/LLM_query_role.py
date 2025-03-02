import asyncio

from typing import List

from tree.node import BaseNode, State

class LLM_QUERY_ROLE():
    def __init__(self):
        pass

    def llm_query_role(self, nodes: List[BaseNode]):
        if len(nodes) == 0:
            return
        self.llm_query(nodes)

    def llm_query_prepare_input(self):
        nodes = self.llm_query_lst
        need_llm_input = []
        if len(nodes) == 0:
            return need_llm_input

        for node in nodes:
            need_llm_input.append({
                "id": self.get_tree_node_tag(node),
                "role": self.config['role']['QUERY_LLM']['name'],
                "need_generate": True,
                "text": node.state.output_text,
                "message": node.state.action_input,
                "generate_config": node.state.thought,
            })

        return need_llm_input

    def llm_query_postprocess_output(self, outputs: dict):
        nodes = self.llm_query_lst
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
                new_state = self.parse_query_LLM(output['text'], response)
                new_node = BaseNode(node_id=node.node_id + f".{len(node.children)}", parent=node, state=new_state, depth=node.depth+1, role=self.config['role']['QUERY_LLM']['name'], plan=node.plan, documents=node.documents, query_history=node.query_history)

                new_node.check_answer(qa_pairs=self.tree.qa_pairs, answers=self.tree.answers, is_asqa=self.tree.is_asqa)
                self.tree.add_solution_nodes(new_node.return_solution_state(node_id=new_node.node_id))
                node.add_child(new_node)
    
    def parse_query_LLM(self, input_text:str, output_text: str) -> State:
        if len(output_text) == 0:
            return State(input_text=input_text, output_text=output_text, is_terminal=True, terminal_reason="Empty Output")
        else:
            return State(input_text=input_text, output_text=output_text, is_terminal=True, action=self.config['role']['QUERY_LLM']['actions']['QUERY_LLM_ACTION'], terminal_reason="End Reason")