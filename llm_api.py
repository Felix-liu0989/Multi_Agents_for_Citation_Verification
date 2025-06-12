import requests
import time
import os, json
from tqdm import tqdm
import traceback
from openai import OpenAI

API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "zhipu": '',
    "anthropic": '',
    'vllm': 'token-abc123',
    'deepseek': "sk-3b9c860b83844a789ea639881a569acc",
    'gemini': "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",
    'qwen' : os.getenv("DASHSCOPE_API_KEY"),
    'llama':"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"
}

API_URLS = {
    'openai': 'https://api.openai.com/v1/chat/completions',
    'zhipu': 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
    "anthropic": 'https://api.anthropic.com/v1/messages',
    'vllm': 'http://127.0.0.1:8000/v1/chat/completions',
    'deepseek': 'https://api.deepseek.com',
    'gemini':"https://openrouter.ai/api/v1",
    'qwen':"https://dashscope.aliyuncs.com/compatible-mode/v1",
    'llama':"https://openrouter.ai/api/v1"
}

def query_llm(messages, model, temperature=1.0, max_new_tokens=1024, stop=None, return_usage=False):
    if 'gpt' in model:
        api_key, api_url = API_KEYS['openai'],None
    elif 'glm' in model:
        api_key, api_url = API_KEYS['zhipu'], API_URLS['zhipu']
    elif 'claude' in model:
        api_key, api_url = API_KEYS['anthropic'], API_URLS['anthropic']
    elif 'deepseek' in model:
        api_key, api_url = API_KEYS['deepseek'], API_URLS['deepseek']
        # print(api_key, api_url)
    elif 'gemini' in model:
        api_key, api_url = API_KEYS['gemini'], API_URLS['gemini']
    else:
        api_key, api_url = API_KEYS['vllm'], API_URLS['vllm']
    tries = 0
    while tries < 5:
        tries += 1
        try:
            client = OpenAI(api_key=api_key, base_url=api_url)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            
            break
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            if "maximum context length" in str(e):
                raise e
            elif "triggering" in str(e):
                return 'Trigger OpenAI\'s content management policy.'
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return None
    
    return resp.choices[0].message.content,resp.usage.to_dict()
    # try:
    #     if 
    #     resp.choices[0].message.content
    #     if 'content' not in resp["choices"][0]["message"] and 'content_filter_results' in resp["choices"][0]:
    #         resp["choices"][0]["message"]["content"] = 'Trigger OpenAI\'s content management policy.'
    #     if return_usage:
    #         return resp["choices"][0]["message"]["content"], resp['usage']
    #     else:
    #         return resp["choices"][0]["message"]["content"]
    # except: 
    #     return None

if __name__ == '__main__':
    model = "deepseek-chat" #"gpt-4o-mini" #"deepseek-chat" # "qwen-plus"
    # 'glm-4-0520'
    prompt = '你是谁'
    msg = [{'role': 'user', 'content': prompt}]
    output = query_llm(msg, model=model, temperature=1, max_new_tokens=1024, stop=None, return_usage=True)
    if isinstance(output, tuple):
        output, usage = output
        print(usage.to_dict())
    else:
        print(output)