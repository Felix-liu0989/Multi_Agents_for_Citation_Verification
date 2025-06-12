from multiprocessing import Pool
import traceback
import re
import jsonlines,json
import os
from llm_api import query_llm
from utils import text_split_by_punctuation,postprocess
from eval_correct import gpt_score_qa,gpt_score_summ,gpt_score_fewshot
from Refine import generate_feedback,optimize_response
with open('D:\Mydesktop\Agent文献\Code\LongCite-main\LongBench-Cite\one_shot_prompt.txt', "r",encoding='utf-8') as fp:
    prompt_format = fp.read()

save_dir = "preds"
os.makedirs(f'{save_dir}/tmp',exist_ok=True)
ipts = json.load(
    open("D:\Mydesktop\Agent文献\Code\LongCite-main\LongBench-Cite\LongBench-Cite.json",encoding='utf-8')
)

cite_model = 'deepseek-chat'
save_name = cite_model.replace('-','_')
fout_path = f'{save_dir}/tmp/{save_name}_0611.jsonl'

if os.path.exists(fout_path):
    with open(fout_path,'r',encoding='utf-8') as f:
        opts = [json.loads(line) for line in f]
else:
    opts = []

s = set(x['idx'] for x in opts)
need_list = [x for x in ipts[0:13] if x['idx'] not in s][:]
print(f'Model: {cite_model}')
print(f'Already predict: {len(opts)} | Remain to predict: {len(need_list)}')
if len(need_list) == 0:
    exit()

dataset2metric = {
    "longbench-chat": gpt_score_fewshot,
    "multifieldqa_en": gpt_score_qa,
    "multifieldqa_zh": gpt_score_qa,
    "multifieldqa": None,
    "hotpotqa": gpt_score_qa,
    "dureader": gpt_score_qa,
    "gov_report": gpt_score_summ,
}


def process(js):
    try:
        context,query = js['context'],js['query']
        sentences = text_split_by_punctuation(context,return_dict = True)
        passage = ""
        # import pdb;pdb.set_trace()
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
        
        # 边生成边引用评估
        from auto_scorer import get_citation_score
        evaluation_result = get_citation_score({
            'query': query,
            'prediction': output,
            'statements': statements_with_citations
        })
        
        
        # 边生成边正确性评估
        dataset = js['dataset']
        ground_truths,  few_shot_scores = js['answer'], js['few_shot_scores']
        del js['few_shot_scores']
        prediction = re.sub(r"<cite>.*?</cite>", "", output, flags=re.DOTALL)
        prediction = prediction.replace('<statement>', '').replace('</statement>', '')
        gpt_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'responses': []}
        correct_score = 0
        for ground_truth in ground_truths:
            correct_score = max(correct_score,dataset2metric[dataset](prediction, ground_truth, query=query, few_shot_scores=few_shot_scores, gpt_usage=gpt_usage))
        js['correct_score'] = correct_score
        js['gpt_usage'] = gpt_usage
        res = {
            'idx': js['idx'],
            'dataset': js['dataset'],
            'query': js['query'],
            'answer': js['answer'],
            'few_shot_scores': few_shot_scores,
            'prediction': output,
            'statements': statements_with_citations,
            'evaluation': {
                'citation_recall': evaluation_result['citation_recall'],
                'citation_precision': evaluation_result['citation_precision'],
                'citation_f1': evaluation_result['citation_f1'],
                'correct_score': correct_score,
                'gpt_usage_for_citation_eval': evaluation_result['gpt_usage'],
                'gpt_usage_for_correctness_eval': gpt_usage
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
    
def process_with_iterations(js,max_iterations = 3,score_threshold = 0.7):
    try:
        context,query = js['context'],js['query']
        sentences = text_split_by_punctuation(context,return_dict = True)
        passage = ""
        # import pdb;pdb.set_trace()
        for i,c in enumerate(sentences):
            start,end = c['start_idx'],c['end_idx']
            assert c['content'] == context[start:end],c
            end = sentences[i+1]['start_idx'] if i < len(sentences) - 1 else len(context)
            passage += f"<C{i}>"+context[start:end]
        
        # 储存每次迭代的结果
        iterations_results = []
        current_output = None
        comprehensive_score = 0 # current
        for iteration in range(max_iterations):
            print(f"Iteration {iteration+1} of {max_iterations}")
            if iteration == 0:
                # 初始生成
                prompt = prompt_format.replace('<<context>>',passage).replace('<<question>>',query)
                msg = [
                    {'role':'system','content':prompt}
                ]
                current_output = query_llm(
                    messages=msg,
                    model=cite_model,
                    temperature = 1,
                    max_new_tokens=1024,
                    return_usage=False
                )
            else:
                # 基于反馈生成优化
                feedback = iterations_results[-1]['feedback']
                current_output = optimize_response(
                    query,
                    current_output,
                    feedback,
                    cite_model,
                    context
                )
        
            if isinstance(current_output,tuple):
                current_output,gpt_usage = current_output
            if current_output.startswith('<statement>'):
                current_output = "<" + current_output
                print(current_output)
        
            statements_with_citations = postprocess(current_output,context,sentences)
        
            # 边生成边引用评估
            from auto_scorer import get_citation_score
            cur_evaluation_result = get_citation_score({
                'query': query,
                'prediction': current_output,
                'statements': statements_with_citations
            })
        
        
            # 边生成边正确性评估
            dataset = js['dataset']
            ground_truths,  few_shot_scores = js['answer'], js['few_shot_scores']
            # del js['few_shot_scores']
            prediction = re.sub(r"<cite>.*?</cite>", "", current_output, flags=re.DOTALL)
            prediction = prediction.replace('<statement>', '').replace('</statement>', '')
            gpt_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'responses': []}
            correct_score = 0
            for ground_truth in ground_truths:
                correct_score = max(correct_score,dataset2metric[dataset](prediction, ground_truth, query=query, few_shot_scores=few_shot_scores, gpt_usage=gpt_usage))
            js['correct_score'] = correct_score
            js['gpt_usage'] = gpt_usage
            res = {
                'idx': js['idx'],
                'dataset': js['dataset'],
                'query': js['query'],
                'answer': js['answer'],
                'few_shot_scores': few_shot_scores,
                'prediction': current_output,
                'statements': statements_with_citations,
                'evaluation': {
                    'citation_recall': cur_evaluation_result['citation_recall'],
                    'citation_precision': cur_evaluation_result['citation_precision'],
                    'citation_f1': cur_evaluation_result['citation_f1'],
                    'correct_score': correct_score,
                    'gpt_usage_for_citation_eval': cur_evaluation_result['gpt_usage'],
                    'gpt_usage_for_correctness_eval': gpt_usage
                }
            }
            
            # 生成反馈
            feedback = generate_feedback(
                query,
                current_output,
                res,
                cite_model
            )
            
            # 记录本次迭代结果
            comprehensive_score = correct_score * 0.5 + cur_evaluation_result['citation_recall'] * 0.5
            iterations_result = {
                'iteration': iteration + 1,
                'feedback': feedback[0],
                'prediction': current_output,
                'evaluation': res['evaluation'],
                'correct_score': correct_score,
                'gpt_usage': gpt_usage,
                'comprehensive_score': comprehensive_score
            }
            iterations_results.append(iterations_result)
            
            print(f"-"*50,flush=True)
            print(f"Iteration {iteration+1} of {max_iterations} is done!")
            print(f"Correct score: {correct_score}")
            print(f"Citation recall: {cur_evaluation_result['citation_recall']}")
            print(f"Citation precision: {cur_evaluation_result['citation_precision']}")
            print(f"Citation f1: {cur_evaluation_result['citation_f1']}")
            print(f"Comprehensive score: {comprehensive_score}")

            if comprehensive_score > score_threshold:
                print(f"达到目标分数{score_threshold}，停止迭代")
                break

            
            if iteration > 0:
                previous_score = iterations_results[-2]['comprehensive_score']
                if comprehensive_score <= previous_score:
                    print(f"当前分数{comprehensive_score}小于等于上一次分数{previous_score}，停止迭代")
                    break
        # 选择最佳结果
        res = {
                'idx': js['idx'],
                'dataset': js['dataset'],
                'query': js['query'],
                'answer': js['answer'],
                'few_shot_scores': few_shot_scores,
                'prediction': current_output,
                'statements': statements_with_citations,
                'evaluation': {
                    'citation_recall': cur_evaluation_result['citation_recall'],
                    'citation_precision': cur_evaluation_result['citation_precision'],
                    'citation_f1': cur_evaluation_result['citation_f1'],
                    'correct_score': correct_score,
                    'gpt_usage_for_citation_eval': cur_evaluation_result['gpt_usage'],
                    'gpt_usage_for_correctness_eval': gpt_usage
                },
                'iterations': iterations_results
            }
        
        best_result = max(iterations_results,key=lambda x:x['comprehensive_score'])
        res['best_result'] = best_result
            
            
            
            
            
            
            #  基于反馈生成优化
            # current_output = optimize_response(
            #     query,
            #     current_output,
            #     feedback,
            #     cite_model,
            #     context
            # )
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
            result = process_with_iterations(item)
            if result:
                print("\n最终结果:")
                print(f"最佳迭代次数: {result['best_result']['iteration']}")
                print(f"最终分数: {result['best_result']['comprehensive_score']:.3f}")
                print(f"最终正确性: {result['best_result']['correct_score']:.3f}")
                print(f"最终召回: {result['best_result']['evaluation']['citation_recall']:.3f}")
                print(f"最终精确: {result['best_result']['evaluation']['citation_precision']:.3f}")
                print(f"最终F1: {result['best_result']['evaluation']['citation_f1']:.3f}")
                # print("")
                # print("\n总体评估结果:")
                # print(f"Recall: {result['evaluation']['citation_recall']:.3f}")
                # print(f"Precision: {result['evaluation']['citation_precision']:.3f}")
                # print(f"F1: {result['evaluation']['citation_f1']:.3f}")
                # print(f"正确性得分: {result['evaluation']['correct_score']:.3f}")
    else:
        # 正常多进程模式
        with Pool(4) as p:
            p.map(process,need_list)

    
        
        
    
            