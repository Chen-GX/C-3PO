import asyncio
import aiohttp

from typing import List

from tree.node import BaseNode, State

from prompt.planning import prompt_llm_planning, prompt_llm_planning_few_shot, instruct_for_each_dataset

from utils import few_shot_random_select

class LLM_PLANNING_ROLE():
    def __init__(self):
        pass

    def llm_planning_role(self, nodes: List[BaseNode]):
        if len(nodes) == 0:
            return
        self.llm_planning(nodes)

    async def async_generate(self, message_lst: List[str], model_url_lst: List[str], n_generate_sample: int):
        async with aiohttp.ClientSession() as session:
            tasks = [self.llm.async_query(msg, model_url, n_generate_sample) for msg, model_url in zip(message_lst, model_url_lst)]
            outputs = await asyncio.gather(*tasks)
        return outputs

    def llm_planning_prepare_input(self):
        nodes = self.planning_lst
        need_llm_input = []
        if len(nodes) == 0:
            return need_llm_input

        for i, node in enumerate(nodes):
            info = {
                "examples": few_shot_random_select(self.few_shot_examples, 'planning', num=self.args.few_num, dict_num=self.args.dict_few_num),
                "question": self.tree.question,
                "dataset_instructions": instruct_for_each_dataset.get(self.args.dataname, ""),
            }
            input_text = prompt_llm_planning_few_shot.format_map(info)
            # model_url = self.args.llm_server_url[i % len(self.args.llm_server_url)]

            message =[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text},
            ]
            need_llm_input.append({
                "id": self.get_tree_node_tag(node),
                "role": self.config['role']['LLM_PLANNING']['name'],
                "need_generate": True,
                "text": input_text,
                "message": message,
                "cache_key": f"{self.tree.question_id}_{self.tree.question}",
                "generate_config": {
                    "n": self.n_plan_sample,
                    "temperature": self.args.plan_temperature,
                    "top_p": self.args.plan_top_p,
                }
            })

        return need_llm_input

    def llm_planning_postprocess_output(self, outputs: dict):
        nodes = self.planning_lst
        if len(nodes) == 0:
            return

        # global dudeplicate due to plan step only in second layer and the nodes on first layer only [Planning]
        output_text = set()
        for node in nodes:
            tag = self.get_tree_node_tag(node)
            
            output = outputs[tag]
            for response in output['outputs']:
                if response in output_text:
                    continue
                output_text.add(response)
                new_state = self.parse_planning(output['text'], response)
                new_node = BaseNode(node_id=node.node_id + f".{len(node.children)}", parent=node, state=new_state, depth=node.depth+1, role=self.config['role']['LLM_PLANNING']['name'], plan=node.plan, documents=node.documents, query_history=node.query_history)
                new_node.update_plan()
                node.add_child(new_node)

    def parse_planning(self, input_text:str, output_text: str) -> State:
        if len(output_text) == 0:
            return State(input_text=input_text, output_text=output_text, is_terminal=True, terminal_reason="Empty Output")
        else:
            return State(input_text=input_text, output_text=output_text, action=self.config['role']['LLM_PLANNING']['actions']['LLM_PLANNING_ACTION'])

        