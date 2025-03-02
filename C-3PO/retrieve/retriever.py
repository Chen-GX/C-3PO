from typing import List

try:
    from retrieve.retrieve_engine import Dense_Retrieve
    from retrieve.search_engine import Search_Engine
except:
    from retrieve_engine import Dense_Retrieve
    from search_engine import Search_Engine

class BasicRAG:
    def __init__(self, args):
        self.args = args
        self.retriever_type = args.retriever_type
        
        if self.retriever_type == "dense":
            self.retriever = Dense_Retrieve(self.args.retrieve_server_url, self.args.retrieve_top_k)
        elif self.retriever_type == "search_engine":
            search_engine_cache_file = self.args.search_engine_cache_file if self.args.search_engine_cache else None
            self.retriever = Search_Engine(self.args.search_engine_url, self.args.search_scene, self.args.retrieve_top_k, search_engine_cache_file)
        else:
            raise NotImplementedError
    

    def retrieve(self, queries: List[str], retrieve_times: List[int]) -> List[List[str]]:
        # length of each query less than args.max_query_length
        if self.retriever_type == "dense":
            passages = self.retriever.retrieve(messages=queries, retrieve_times=retrieve_times)
            return passages
        elif self.retriever_type == "search_engine":
            passages = self.retriever.retrieve(messages=queries, retrieve_times=retrieve_times)
            return passages
        else:
            raise NotImplementedError


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--retriever_type', type=str, default="dense", choices=["dense", "BM25"])
    parser.add_argument('--retrieve_server_url', type=str, default="http://10.32.18.155:35004/search")
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--max_query_length', type=int, default=100)

    args = parser.parse_args()
    rag = BasicRAG(args)
    rag.retrieve(["What is the capital of China?"])
    print()