import os
import os.path as osp
import json
import random
import requests
import time

from time import sleep
from openai import OpenAI
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)

from pebble import ThreadPool
import threading
from concurrent.futures import TimeoutError
from typing import List, Union
from transformers import AutoTokenizer
from filelock import FileLock

from utils import load_json

def read_and_check_ip(file_path):
    lock = FileLock(f"{file_path}.lock")
    while True:
        try:
            with lock:
                with open(file_path, "r") as f:
                    ip_json = json.load(f)
            break
        except Exception as e:
            logger.error(f"Error reading ip file {file_path}: {str(e)}")
            sleep(1)
    
    healthy_urls = []
    with requests.Session() as session:
        for url in ip_json.keys():
            try:
                health_url = url.strip("v1") + "health"
                response = session.get(health_url, timeout=0.2)
                if response.status_code == 200:
                    healthy_urls.append(url)
                    ip_json[url] = "True"
                else:
                    ip_json[url] = "False"
                    logger.warning(f"Unhealthy response from {health_url}: Status code {response.status_code}")
            except requests.exceptions.RequestException as e:
                ip_json[url] = "False"
                logger.error(f"Error url {url}: {str(e)}")

    for key in ip_json.keys():
        if key in healthy_urls:
            ip_json[key] = "True"
        else:
            ip_json[key] = "False"
            
    with lock:
        with open(file_path, "w") as f:
            json.dump(ip_json, f, indent=2)
        
    return healthy_urls

class LLM_SERVER():
    def __init__(self, args, llm=None, sampling_params=None):
        self.llm_server_type = args.llm_server_type
        self.wo_llm = args.wo_llm
        if not self.wo_llm:
            if self.llm_server_type == "online":
                self.llm_clients = [
                    OpenAI(api_key=args.llm_api_key, base_url=url)
                    for url in args.llm_server_url
                ]
                self.current_client = 0  # 用于轮询
                self.base_concurrency = args.online_concurrency
                self.concurrency = self.base_concurrency * len(self.llm_clients)
                self.model_name = args.llm_name
                self.lock = threading.Lock()  # 添加线程锁
                self.retry_times = 10

            elif self.llm_server_type == "offline":
                self.llm, self.sampling_params = llm, sampling_params
                self.tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
                self.backend = args.backend

        self.use_planning_cache = args.use_planning_cache
        if self.use_planning_cache:
            self.cache = {}

            if "ppo_train" in args.data_file_name:
                cache = load_json(osp.join(args.cache_dir, args.dataname, "ppo_train.json"))
            elif "train" in args.data_file_name:
                cache = load_json(osp.join(args.cache_dir, args.dataname, "train.json"))
            else:
                cache = load_json(osp.join(args.cache_dir, args.dataname, "test.json"))
            self.cache.update(cache)
            logger.info("use cache for planning")
    
    def check_cache(self, data):
        for item in data:
            key = f"{item['id']}_{item['question']}"
            if key not in self.cache:
                assert False, f"key {key} not in cache"
        logger.info("cache check success")

    def get_next_client(self):
        client = self.llm_clients[self.current_client]
        self.current_client = (self.current_client + 1) % len(self.llm_clients)
        return client

    def llm_generate(self, query_list, max_retries=1e8, retry_delay=1):
        index = query_list["index"]
        messages = query_list["messages"]
        sampling_params = query_list["sampling_params"]
        # 为每个请求获取下一个可用的客户端
        client = self.get_next_client()
        
        outputs = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            n=sampling_params['n'],
            temperature=sampling_params['temperature'],
            top_p=sampling_params['top_p'],
            max_tokens=sampling_params.get('max_new_tokens', 1024),
        )
        return {"index": index, "outputs": outputs}


    def generate(self, prompt_dict: List[dict]):
        # planning cache for faster generation
        if self.use_planning_cache:
            for item in prompt_dict:
                if item['need_generate'] and item['role'] == "LLM_PLANNING":
                    if item['cache_key'] in self.cache:
                        item['outputs'] = random.sample(self.cache[item['cache_key']], min(item['generate_config']['n'], len(self.cache[item['cache_key']])))
                        item['need_generate'] = False

        need_generate_prompt = [item for item in prompt_dict if item['need_generate']]

        if len(need_generate_prompt) == 0:
            return prompt_dict

        if self.wo_llm:
            for item in need_generate_prompt:
                item['outputs'] = [""] * item['generate_config']['n']
            return prompt_dict
        
        if self.llm_server_type == "online":
            query_list = [{"index": i, "messages": item['message'], "sampling_params": item['generate_config']} for i, item in enumerate(need_generate_prompt)]
            
            max_workers = min(len(query_list), self.concurrency)  # 每个URL允许多个并发
            with ThreadPool(max_workers=max_workers) as pool:
                future = pool.map(self.llm_generate, query_list) # , timeout=180
                # outputs = list(future.result())
                outputs = list(tqdm(future.result(), total=len(query_list), desc="LLM Server"))
                
            assert len(outputs) == len(need_generate_prompt)
            for item, output in zip(need_generate_prompt, outputs):
                response = [o.message.content for o in output['outputs'].choices]
                item['outputs'] = response
        elif self.llm_server_type == "offline":
            assert False, "offline llm not implemented"
            prompts = [
                self.tokenizer.apply_chat_template(
                    item['message'],
                    tokenize=False,
                    add_generation_prompt=True
                ) for item in need_generate_prompt
            ]
            n = max([item['generate_config']['n'] for item in need_generate_prompt])
            
            if self.backend == "vllm":
                self.sampling_params.n = n
                self.sampling_params.best_of = n
                outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
                
                for item, output in zip(need_generate_prompt, outputs):
                    response = [item.text for item in output.outputs]
                    item['outputs'] = response[:item['generate_config']['n']]
                
            elif self.backend == "sglang":
                self.sampling_params["n"] = n
                outputs = self.llm.generate(prompts, self.sampling_params)
                # outputs is [{}, {}]
                # we need split outpus by n to each item
                assert len(outputs) == n * len(need_generate_prompt)
                outputs = [outputs[i:i+n] for i in range(0, len(outputs), n)]
                for item, output in zip(need_generate_prompt, outputs):
                    response = [item['text'] for item in output]
                    item['outputs'] = response

        return prompt_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_server_url', type=str, default="http://10.32.8.56:8000/v1,http://10.32.8.56:8001/v1")
    parser.add_argument('--openai_api_key', type=str, default="EMPTY")
    parser.add_argument('--llm_name', type=str, default="qwen2-72b-instruct")
    args = parser.parse_args()
    args.llm_server_url = args.llm_server_url.split(",")
    llm = LLM_SERVER(args)
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": "Tell me something about large language models."},
    ]
    message_lst = [messages] * 2
    model_url_lst = args.llm_server_url
    import aiohttp
    import asyncio

    async def async_generate(message_lst, model_url_lst, n_generate_sample):
        async with aiohttp.ClientSession() as session:
            tasks = [llm.async_query(msg, model_url, 5) for msg, model_url in zip(message_lst, model_url_lst)]
            outputs = await asyncio.gather(*tasks)
        return outputs

    outputs = asyncio.run(async_generate(message_lst, model_url_lst, 5))
    print()

