import requests
import json
import time

def use_flask():  # 传入待分析的文本，与token
    # url = "http://10.140.0.136:35000/search"
    # url = "http://10.32.25.199:35004/search"
    url = "http://10.32.25.199:35002/search"
    headers = {'Content-Type': 'application/json'}  # 设置请求头
    data = json.dumps({
        # 'queries': ["The director of the film Swarg Narak"],  # , "what's the weather today?"   * 128
        # 'queries': ["Director of Detective Chinatown 2 birthplace"],
        'queries': ["""the Brooklyn Dodgers and the Boston Braves on 11 August 11 1951"""],  # Warren Beatty  Who is the director of Bulworth?
        "n_docs": 10
    })
    begin_time = time.time()
    response = requests.post(url, headers=headers, data=data)  # 发送POST请求
    print(time.time() - begin_time)
    if response.status_code == 200:
        return response.status_code, response.json()  # 状态码，返回JSON对象(这里是结果列表)
    else:
        return response.status_code, response.raise_for_status()  # 如果响应状态码不是200, 抛出异常



if __name__ == '__main__':
    data = use_flask()
    print(data)
