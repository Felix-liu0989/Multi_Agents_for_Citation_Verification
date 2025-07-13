from multiprocessing import Pool
import traceback
import re
import jsonlines,json
import os
from llm_api import query_llm
from utils import text_split_by_punctuation,postprocess


with open('D:\Mydesktop\Agent文献\Code\LongCite-main\LongBench-Cite\one_shot_prompt.txt', "r",encoding='utf-8') as fp:
    prompt_format = fp.read()

save_dir = "preds"
os.makedirs(f'{save_dir}/tmp',exist_ok=True)
ipts = json.load(
    open("D:\Mydesktop\Agent文献\Code\LongCite-main\LongBench-Cite\LongBench-Cite.json",encoding='utf-8')
)

cite_model = 'deepseek-chat'
save_name = cite_model.replace('-','_')
fout_path = f'{save_dir}/tmp/{save_name}.jsonl'

if os.path.exists(fout_path):
    with open(fout_path,'r',encoding='utf-8') as f:
        opts = [json.loads(line) for line in f]
else:
    opts = []

s = set(x['idx'] for x in opts)
need_list = [x for x in ipts[:3] if x['idx'] not in s][:]
print(f'Model: {cite_model}')
print(f'Already predict: {len(opts)} | Remain to predict: {len(need_list)}')
if len(need_list) == 0:
    exit()


def process(js):
    try:
        context,query = js['context'],js['query']
        sentences = text_split_by_punctuation(context,return_dict = True)
        passage = ""
        for i,c in enumerate(sentences):
            start,end = c['start_idx'],c['end_idx']
            assert c['content'] == context[start:end],c
            end = sentences[i+1]['start_idx'] if i < len(sentences) - 1 else len(context)
            passage += f"<C{i}>"+context[start:end]
            
        prompt = prompt_format.replace('<<context>>',passage).replace('<<question>>',query)
        msg = [
            {'role':'system','content':prompt}
        ]
        output = query_llm(
            messages=msg,
            model=cite_model,
            temperature = 1,
            max_new_tokens=1024,
            return_usage=False
        )
        
        print(output)
        if output[0].startswith('<statement>'):
            output = "<" + output[0]
            print(output)
        
        statements_with_citations = postprocess(output,context,sentences)
        
        # 边生成边评估
        from auto_scorer import get_citation_score
        evaluation_result = get_citation_score({
            'query': query,
            'prediction': output,
            'statements': statements_with_citations
        })
        
        res = {
            'idx': js['idx'],
            'dataset': js['dataset'],
            'query': js['query'],
            'answer': js['answer'],
            'few_shot_scores': js['few_shot_scores'],
            'prediction': output,
            'statements': statements_with_citations,
            'evaluation': {
                'citation_recall': evaluation_result['citation_recall'],
                'citation_precision': evaluation_result['citation_precision'],
                'citation_f1': evaluation_result['citation_f1'],
                'gpt_usage': evaluation_result['gpt_usage']
            }
        }
        
        with jsonlines.open(fout_path,mode='a') as writer:
            writer.write(res)
        
        return res
    except:
        print(js['idx'])
        print(query)
        traceback.print_exc()
        print('-'*100)
        return None
    

if __name__ == "__main__":
    # 调试模式
    DEBUG = True
    
    if DEBUG:
        # 单进程调试模式
        for item in need_list[:]:  # 处理所有样本
            print(f"\n处理样本 {item['idx']}:")
            print(f"问题: {item['query']}")
            result = process(item)
            if result:
                print("\n总体评估结果:")
                print(f"Recall: {result['evaluation']['citation_recall']:.3f}")
                print(f"Precision: {result['evaluation']['citation_precision']:.3f}")
                print(f"F1: {result['evaluation']['citation_f1']:.3f}")
    else:
        # 正常多进程模式
        with Pool(4) as p:
            p.map(process,need_list)

    
        
        
    
            