import requests
from typing import List

import logging

logger = logging.getLogger(__name__)

timeout_duration = 360

class Dense_Retrieve():
    def __init__(self, server_url, top_k):
        self.server_url = server_url  # TODO: 或许能够通过多个url进行负载均衡
        self.headers = {"Content-Type": "application/json"}
        self.top_k = top_k
    
    def retrieve(self, messages: List[str], retrieve_times: List[int]) -> List[List[str]]:
        n_docs = (max(retrieve_times) + 1) * self.top_k
        data = {
            "queries": messages,
            "n_docs": n_docs,
        }
        try:
            response = requests.post(self.server_url, json=data, headers=self.headers, timeout=timeout_duration)
        except requests.ConnectionError as e:
            logger.info(f"Retrieve Connection Error: {e}")
            raise RuntimeError("Retrieve Connection Error") from e
        except requests.Timeout as e:
            logger.info(f"Retrieve Timeout: {e}")
            raise RuntimeError("Retrieve Timeout") from e
        except requests.HTTPError as e:
            logger.info(f"Request Error: {e}")
            raise RuntimeError(f"HTTP Error: {e.response.status_code}") from e
        except requests.RequestException as e:
            logger.info(f"Request Error: {e}")
            raise RuntimeError("Request Error") from e
        else:
            selected_passages = []
            for r_time, passages in zip(retrieve_times, response.json()):
                selected_passages.append(passages[r_time * self.top_k:self.top_k * (r_time + 1)])
            return selected_passages