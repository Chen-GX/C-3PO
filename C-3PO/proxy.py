import os
from time import sleep
from openai import OpenAI
from typing import List
from pebble import ProcessPool, ThreadPool

from utils import load_agent

import logging

logger = logging.getLogger(__name__)

def load_proxy(args):
    available_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(',')
    logger.info(f"available_gpus: {available_gpus}, NUM_GPUS: {len(available_gpus)}")
    proxy = Proxy(args, num_gpus=len(available_gpus))
    return proxy

class Proxy():

    def __init__(
        self,
        args,
        num_gpus: int = 1,
    ):
        self.backend = args.backend
        self.model_type = args.model_type

        if self.model_type == "proxy":
            self.llm, self.sampling_params, self.server_process = load_agent(args, num_gpus)
            self.concurrency = args.proxy_concurrency
            self.model_name = "default"

        elif self.model_type == "gpt":
            self.model_name = args.proxy_model_name
            self.temperature = args.temperature
            self.top_p = args.top_p
            self.max_tokens = args.max_tokens
            self.concurrency = args.gpt_proxy_concurrency
            self.llm = OpenAI(
                api_key=args.openai_api_key,
                base_url=args.proxy_url,
            )
            self.sampling_params = sampling_params = {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_new_tokens": args.max_tokens,
            }

        else:
            raise NotImplementedError

    def sglang_generate(self, query_list):
        index = query_list.get("index", 0)
        messages = query_list["messages"]
        sampling_params = query_list["sampling_params"]
        outputs = self.llm.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    n=sampling_params['n'],
                    temperature=sampling_params['temperature'],
                    top_p=sampling_params['top_p'],
                    max_tokens=sampling_params['max_new_tokens'],
                )
        return {"index": index, "outputs": outputs}

    def generate(self, prompt_dict: List[dict], n_generate_sample: int):
        need_generate_prompt = [item for item in prompt_dict if item['need_generate']]
        if len(need_generate_prompt) == 0:
            return prompt_dict

        prompt = [item['template_text'] for item in need_generate_prompt]

        if self.model_type == "proxy" and self.backend == "vllm":
            self.sampling_params.n = n_generate_sample
            self.sampling_params.best_of = n_generate_sample
            outputs = self.llm.generate(prompt, self.sampling_params, use_tqdm=True)
            
            for item, output in zip(need_generate_prompt, outputs):
                response = [item.text for item in output.outputs]
                item['outputs'] = response
            return prompt_dict

        else:
            self.sampling_params["n"] = n_generate_sample
            query_list = [{"index": i, "messages": p, "sampling_params": self.sampling_params} for i, p in enumerate(prompt)]
            with ThreadPool(max_workers=self.concurrency) as pool:
                future = pool.map(self.sglang_generate, query_list)
                outputs = list(future.result())
            sorted_outputs = sorted(outputs, key=lambda x: x["index"])

            assert len(sorted_outputs) == len(need_generate_prompt)
            for item, output in zip(need_generate_prompt, sorted_outputs):
                response = [o.message.content for o in output['outputs'].choices]
                item['outputs'] = response
            return prompt_dict