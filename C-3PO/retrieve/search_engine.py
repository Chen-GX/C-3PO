import os
import os.path as osp
import json
import requests
from time import sleep
from typing import List
from pebble import ThreadPool
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)

timeout_duration = 360

class Search_Engine():
    def __init__(self, server_url, search_scene, top_k, search_engine_cache_file=None):
        self.server_url = server_url  # TODO: 或许能够通过多个url进行负载均衡
        self.search_scene = search_scene
        self.headers = {
            'Content-Type': 'application/json',
            '__d_head_qto': '5000',
            '__d_head_app': 'Test',
            "Authorization": "Bearer {your_key}",
        }
        self.top_k = top_k
        self.search_engine_cache_file = search_engine_cache_file
        self.retry_times = 10
        if search_engine_cache_file is not None:
            self.search_engine_cache_file = osp.join(search_engine_cache_file, search_scene, "cache_search.json")
            if osp.exists(self.search_engine_cache_file):
                with open(self.search_engine_cache_file, "r") as f:
                    self.cache = json.load(f)
            else:
                self.cache = {}

    def save_cache(self):
        if self.search_engine_cache_file is not None:
            os.makedirs(osp.dirname(self.search_engine_cache_file), exist_ok=True)
            new_dict = {}
            for key, value in self.cache.items():
                if value['success'] == True:
                    new_dict[key] = value
            with open(self.search_engine_cache_file, "w") as f:
                json.dump(self.cache, f, indent=2)

    def single_query(self, payload):
        for _ in range(self.retry_times):
            response = requests.post(self.server_url, headers=self.headers, json=payload).json()
            sleep(0.5)
            if response['status'] == 0:
                return {json.dumps(payload): response}

        raise ValueError(f"{response['message']}")
    
    def retrieve(self, messages: List[str], retrieve_times: List[int]) -> List[List[str]]:
        query_list, key_list = [], []
        for message, r_time in zip(messages, retrieve_times):
            n_docs = (r_time + 1) * self.top_k
            payload = {
                "rid": "",
                "scene": self.search_scene,
                "uq": message,
                "debug": False,
                "fields": [],
                "page": 1,
                "rows": n_docs,
                "customConfigInfo": {
                    "readpage": False,
                },
            }
            key = json.dumps(payload)
            key_list.append(key)
            if key not in self.cache:
                query_list.append(payload)

        with ThreadPool(max_workers=1) as pool:
            future = pool.map(self.single_query, query_list)
            outputs = list(tqdm(
                future.result(),
                total=len(query_list),
                desc="Searching queries"
            ))

        outputs = {key: value for item in outputs for key, value in item.items()}
        selected_passages = []
        for r_time, key in zip(retrieve_times, key_list):
            if key in self.cache:
                response = self.cache[key]
            else:
                response = outputs[key]
                if response['success'] == True:
                    self.cache[key] = response
            passages = [{"id": item['_id'], "text": item['snippet'], "title": item['title']} for item in response['data']['docs']]
            selected_passages.append(passages[r_time * self.top_k:self.top_k * (r_time + 1)])
        return selected_passages

if __name__ == "__main__":
    url = ""
    search_scene = ""
    top_k = 10
    search_engine_cache_file = "path to cache_search"
    search_engine = Search_Engine(url, search_scene, top_k, search_engine_cache_file)
    queries = ["who is the director of Mera Naam Joker"]
    retrieve_times = [0]
    passages = search_engine.retrieve(queries, retrieve_times)
    print(passages)
    search_engine.save_cache()