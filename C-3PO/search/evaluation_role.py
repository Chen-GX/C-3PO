
import yaml
import asyncio

from typing import List, Union

from prompt.evaluation import evaluation_prompt, evaluation_prompt_few_shot

from metrics import get_item_metrics

from utils import few_shot_random_select

from utils import load_few_shot


class EVALUATION_ROLE():
    def __init__(self, args, llm, sampling_params=None):
        if args.only_eval_answer:
            self.args = args  
            self.llm = llm
            self.sampling_params = sampling_params
            self.invalid_message =[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "hi"},
            ]

    def evluation_prepare_input(self):
        need_proxy_input = []
        for idx, solution in enumerate(self.save_tree_info['solution_nodes']):
            infos = {
                "question": self.save_tree_info['question'],
                "true_answer": str(self.save_tree_info['answers']),
                "long_answer": self.save_tree_info['tree'][solution['node_id']]['state']['output_text'],
            }
            input_text = evaluation_prompt.format_map(infos)
            message =[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text},
            ]
            need_proxy_input.append({
                "id": self.get_extract_answer_tag(idx),
                "role": "evaluate_answer",
                "need_generate": True,
                "text": input_text,
                "message": message,
                "generate_config": {
                    "n": self.n_answer_sample,
                    "temperature": self.args.answer_temperature,
                    "top_p": self.args.answer_top_p,
                }
            })

        return need_proxy_input

    def evluation_postprocess_output(self, outputs: dict):
        for idx, solution in enumerate(self.save_tree_info['solution_nodes']):
            tag = self.get_extract_answer_tag(idx)
            output = outputs[tag]
            output_text = set()
            for response in output['outputs']:
                if response in output_text:
                    continue

                answer_status = 1 if "true" in response.lower() else 0
                self.save_tree_info['tree'][solution['node_id']]['state']['eval_response'] = response
                self.save_tree_info['tree'][solution['node_id']]['state']['eval_status'] = answer_status

        for item in self.save_tree_info['solution_nodes']:
            item['eval_status'] = self.save_tree_info['tree'][item['node_id']]['state']['eval_status']

    def evaluation_off_input(self, save_tree_info):
        # data_item 是 self.tree.save_tree()
        message_lst = []
        assert len(save_tree_info['solution_nodes']) <=1, "only support one solution node"
        for i, solution in enumerate(save_tree_info['solution_nodes']):
            infos = {
                "question": save_tree_info['question'],
                "true_answer": str(save_tree_info['answers']),
                "long_answer": save_tree_info['tree'][solution['node_id']]['state']['output_text'],
            }
            input_text = evaluation_prompt.format_map(infos)
            message =[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text},
            ]
            message_lst.append(message)
        
        return message_lst if len(message_lst) > 0 else [self.invalid_message]
    
    def evaluation_off_output(self, outputs, save_tree_info):
        for solution, output in zip(save_tree_info['solution_nodes'], outputs):
            if self.args.only_eval_answer and self.args.model_type == "proxy":
                if self.args.backend == "vllm":
                    response = output.outputs[0].text
                elif self.args.backend == "sglang":
                    response = output['outputs'].choices[0].message.content
            else:
                response = output.choices[0].message.content

            answer_status = 1 if "true" in response.lower() else 0
            save_tree_info['tree'][solution['node_id']]['state']['eval_response'] = response
            save_tree_info['tree'][solution['node_id']]['state']['eval_status'] = answer_status

        for item in save_tree_info['solution_nodes']:
            item['eval_status'] = save_tree_info['tree'][item['node_id']]['state']['eval_status']

        return save_tree_info

    def evaluation_off_generate(self, query_list):
        index = query_list.get("index", 0)
        messages = query_list["messages"]
        sampling_params = query_list["sampling_params"]

        outputs = self.llm.chat.completions.create(
                    model="default",
                    messages=messages,
                    n=sampling_params['n'],
                    temperature=sampling_params['temperature'],
                    top_p=sampling_params['top_p'],
                    max_tokens=sampling_params['max_new_tokens'],
                )
        return {"index": index, "outputs": outputs}



