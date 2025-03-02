import os, sys
os.chdir(sys.path[0])
sys.path.append("..")

import yaml
from types import SimpleNamespace

from passage_retrieval import Retriever
from flask import Flask, request, jsonify

app = Flask(__name__)

# 将配置字典转换为一个args对象
def dict_to_simplenamespace(d: dict) -> SimpleNamespace:
    namespace = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            value = dict_to_simplenamespace(value)
        setattr(namespace, key, value)
    return namespace


# 加载参数
with open("wiki18_config.yaml", 'r', encoding='utf-8') as fin:
    configs_dict = yaml.load(fin, Loader=yaml.FullLoader)

configs = dict_to_simplenamespace(configs_dict)

# 预加载的模型
retriever = Retriever(configs)
retriever.setup_retriever()
print("Retriever setup finished.")

@app.route('/search', methods=['POST'])
def search():
    # 获取请求中的查询参数
    args = request.get_json()
    queries = args.get('queries', [])
    top_n = args.get('n_docs', 10)
    retrieved_documents = retriever.search_document(queries, top_n)
    
    # 返回JSON格式的响应
    return jsonify(retrieved_documents)

if __name__ == '__main__':
    # 启动Flask服务器，host='0.0.0.0'使服务器对外可见
    app.run(host='0.0.0.0', port=35004, debug=False)
