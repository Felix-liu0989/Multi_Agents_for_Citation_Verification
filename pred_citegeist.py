from multiprocessing import Pool
import traceback
import re
import jsonlines,json
import os
from llm_api import query_llm
from utils import academic_text_split_by_punctuation,postprocess
from eval_correct import gpt_score_qa,gpt_score_summ,gpt_score_fewshot,gpt_score_academic_instruction
from Refine import generate_feedback,optimize_response
from src.citegeist import Generator
from src.evaluation.agents.judge import Judge
with open('one_shot_prompt_1.txt', "r",encoding='utf-8') as fp:
    prompt_format = fp.read()

save_dir = "preds"
os.makedirs(f'{save_dir}/tmp',exist_ok=True)
ipts = json.load(
    open("datasets/NUDTRWG.json",encoding='utf-8')
)
# "scholar_copilot_eval_data_Remove_Duplicates.json"
cite_model = 'deepseek-chat'
save_name = cite_model.replace('-','_')
fout_path = f'{save_dir}/tmp/{save_name}_0626.jsonl'

if os.path.exists(fout_path):
    with open(fout_path,'r',encoding='utf-8') as f:
        opts = [json.loads(line) for line in f]
else:
    opts = []

s = set(x['idx'] for x in opts)
need_list = [x for x in ipts[0:1] if x['idx'] not in s][:]
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
    "scholar_copilot":gpt_score_academic_instruction
}



def process(js):
    try:
        context = ""
        bib_info = js["bib_info"]
        cite_keys = bib_info.keys()
        for cite_key in cite_keys:
            cite = bib_info[cite_key]
            pattern = r'<\|cite_(\d+)\|>'  

            match = re.search(pattern, cite)
            if match:
                n = match.group(1)  
            else:
                print("未找到匹配的数字")
            abstract = f"<|cite_{n}|>" + cite[0]["abstract"] + f"</|cite_{n}|>"
            context += abstract
            
        
        query = js['abstract']
        
        sentences = academic_text_split_by_punctuation(context,bib_info,num,return_dict = True)
        title = sentences[0]['title']
        title = title.replace('{','')
        passage = f"1. {title} abstract:\n"
        idx = 1
        # import pdb;pdb.set_trace()
        for i,c in enumerate(sentences):
            start,end = c['start_idx'],c['end_idx']
            assert c['content'] == context[start:end],c
            end = sentences[i+1]['start_idx'] if i < len(sentences) - 1 else len(context)
            
            passage += f"<C{i}>"+c['content']
            if c['title'].replace('{','') != title:
                title = c['title']
                title = title.replace('{','')
                passage += f"\n\n{idx+1}. {title} abstract:\n"
                passage += f"<C{i}>"+c['content']
                
                
            
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
        ground_truth = js['paper']
        title = js['title']
        abstract = js['abstract']
        prediction = re.sub(r"<cite>.*?</cite>", "", output, flags=re.DOTALL)
        prediction = prediction.replace('<statement>', '').replace('</statement>', '')
        gpt_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'responses': []}
        correct_score = 0
        score = dataset2metric[dataset](title,abstract,prediction, ground_truth, query=query, gpt_usage=gpt_usage)
        total_score = score['total']
        
        correct_score = max(correct_score,total_score)
        detailed_score = {
            'relevance': score['relevance'],
            'coherence': score['coherence'],
            'academic': score['academic'],
            'completeness': score['completeness'],
            'innovation': score['innovation'],
            'total': total_score
        }
        js['correct_score'] = detailed_score
        js['gpt_usage'] = gpt_usage
        print(correct_score)
        res = {
            'idx': js['idx'],
            'dataset': js['dataset'],
            'abstract': js['abstract'],
            'ground_truth': js['paper'],
            'prediction': output,
            'statements': statements_with_citations,
            'evaluation': {
                'citation_recall': evaluation_result['citation_recall'],
                'citation_precision': evaluation_result['citation_precision'],
                'citation_f1': evaluation_result['citation_f1'],
                'correct_score': int(correct_score)/25,
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
    
def process_with_iterations_v2(js):
    
    generator = Generator(
        llm_provider="gemini",  # Choice of: "azure" (OpenAI Studio), "anthropic", "gemini", "mistral", and "openai"
        api_key=os.environ.get("OPENROUTER_API_KEY"), # Here, you will need to set the respective API key
        model_name="google/gemini-2.5-flash-preview-05-20", # Choose the model that the provider supports
        database_uri=os.environ.get("MILVUS_URI"),  # Set the path (local) / url (remote) for the Milvus DB connection
        database_token=os.environ.get("MILVUS_TOKEN"),  # Optionally, also set the access token (you DON'T need to set this when using the locally hosted Milvus Database)
    )
    
    # Define input abstract and breadth (5-20), depth (1-5), and diversity (0.0-1.0) parameters.
    abstract = js['abstract']
    breadth = 10
    depth = 2
    diversity = 0.0
    
    # 生成相关工作
    result = generator.generate_related_work(abstract, breadth, depth, diversity)
    
    # ========== 添加评估环节 ==========
    # 初始化评估客户端
    from src.citegeist.utils.llm_clients.gemini_client import GeminiClient
    from src.citegeist.utils.helpers import load_api_key
    from src.citegeist.utils.prompts import (
        generate_relevance_evaluation_prompt,
        generate_win_rate_evaluation_prompt,
        generate_related_work_score_prompt,
    )
    from src.citegeist.utils.citations import get_arxiv_abstract
    
    # 初始化Azure客户端用于评估
    evaluation_client = GeminiClient(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        model_name="google/gemini-2.5-flash-preview-05-20",
    )
    
    # 1. 评估生成的相关工作与原始论文的相关性
    if 'related_works' in js and js['related_works']:
        # 与原始相关工作对比
        win_rate_prompt, order = generate_win_rate_evaluation_prompt(
            source_abstract=abstract,
            source_related_work=js['related_works'],
            target_related_work=result['related_works']
        )
        
        win_rate_response = evaluation_client.get_completion(
            win_rate_prompt
        )
        
        # 解析评估结果
        selected_winner_original_vs_generated = ""
        if win_rate_response == "Section A":
            selected_winner_original_vs_generated = order[0]
        elif win_rate_response == "Section B":
            selected_winner_original_vs_generated = order[1]
        elif win_rate_response == "Tie":
            selected_winner_original_vs_generated = "tie"
    else:
        selected_winner_original_vs_generated = "no_original_work"
    
    # 2. 评估生成的相关工作质量
    related_work_score_prompt = generate_related_work_score_prompt(
        source_abstract=abstract,
        related_work=result['related_works']
    )
    related_work_score_response = evaluation_client.get_completion(
        related_work_score_prompt
    )
    related_work_score = int(related_work_score_response)/10
    
    
    # 2. 评估生成的相关工作与GPT生成的相关工作的对比
    # if 'related_works_gpt4o_mini' in js and js['related_works_gpt4o_mini']:
    #     win_rate_prompt2, order2 = generate_win_rate_evaluation_prompt(
    #         source_abstract=abstract,
    #         source_related_work=result['related_works'],
    #         target_related_work=js['related_works_gpt4o_mini']
    #     )
        
    #     win_rate_response2 = evaluation_client.get_completions(
    #         win_rate_prompt2,
    #         os.getenv("AZURE_PROMPTING_MODEL_VERSION")
    #     )
        
    #     # 解析评估结果
    #     selected_winner_generated_vs_gpt = ""
    #     if win_rate_response2 == "Section A":
    #         selected_winner_generated_vs_gpt = order2[0]
    #     elif win_rate_response2 == "Section B":
    #         selected_winner_generated_vs_gpt = order2[1]
    #     elif win_rate_response2 == "Tie":
    #         selected_winner_generated_vs_gpt = "tie"
    # else:
    #     selected_winner_generated_vs_gpt = "no_gpt_work"
    judge = Judge(model="google/gemini-2.5-flash-preview-05-20")
    result = judge.citation_quality_author_year(result)
    
    # 3. 评估引用质量
    
    citation_count = len(result['citations']) if result['citations'] else 0
    
    # 4. 计算综合评分
    # 基于评估结果计算综合评分
    evaluation_score = 0.0
    evaluation_score += related_work_score
    # 如果我们的方法赢了原始工作，加分
    if selected_winner_original_vs_generated == "generated":
        evaluation_score += 0.4
    elif selected_winner_original_vs_generated == "tie":
        evaluation_score += 0.2
    
    # 如果我们的方法赢了GPT，加分
    # if selected_winner_generated_vs_gpt == "generated":
    #     evaluation_score += 0.4
    # elif selected_winner_generated_vs_gpt == "tie":
    #     evaluation_score += 0.2
    
    # 基于引用数量加分（适度引用）
    if 3 <= citation_count <= 8:
        evaluation_score += 0.2
    elif citation_count > 8:
        evaluation_score += 0.1
    
    # 将评估结果添加到返回结果中
    result['evaluation'] = {
        'original_vs_generated': selected_winner_original_vs_generated,
        'related_work_score': related_work_score,
        #'generated_vs_gpt': selected_winner_generated_vs_gpt,
        'citation_count': citation_count,
        'evaluation_score': evaluation_score,
        'win_rate_prompt_original': win_rate_prompt if 'related_works' in js and js['related_works'] else None,
        # 'win_rate_prompt_gpt': win_rate_prompt2 if 'related_works_gpt4o_mini' in js and js['related_works_gpt4o_mini'] else None,
        'win_rate_response_original': win_rate_response if 'related_works' in js and js['related_works'] else None,
        # 'win_rate_response_gpt': win_rate_response2 if 'related_works_gpt4o_mini' in js and js['related_works_gpt4o_mini'] else None
    }
    
    print(f"=== 评估结果 ===")
    print(f"与原始相关工作对比: {selected_winner_original_vs_generated}")
    # print(f"与GPT生成相关工作对比: {selected_winner_generated_vs_gpt}")
    print(f"引用数量: {citation_count}")
    print(f"综合评分: {evaluation_score:.2f}")
    print(f"==================")
    
    return result
    
def process_with_iterations(js,max_iterations = 3,score_threshold = 0.7):
    try:
        context = ""
        bib_info = js["bib_info"]
        cite_keys = bib_info.keys()
        for cite_key in cite_keys:
            cite = bib_info[cite_key]
            pattern = r'<\|cite_(\d+)\|>'  

            match = re.search(pattern, cite_key)
            if match:
                n = match.group(1)  
            else:
                print("未找到匹配的数字")
            abstract = ""
            for c in cite:
                abstract += f"<|cite_{n}|>" + c["abstract"] + f"</|cite_{n}|>"
                
            context += abstract
            
        
            
        query = js['abstract']
        
        sentences = academic_text_split_by_punctuation(context,bib_info,return_dict = True)
        
            
        title = sentences[0]['title']
        title = title.replace('{','')
        passage = f"1. {title} abstract:\n"
        idx = 1
        # import pdb;pdb.set_trace()
        for i,c in enumerate(sentences):
            start,end = c['start_idx'],c['end_idx']
            assert c['content'] == context[start:end],c
            end = sentences[i+1]['start_idx'] if i < len(sentences) - 1 else len(context)
            passage += f"<C{i}>"+c['content']
            if c['title'].replace('{','') != title:
                title = c['title']
                title = title.replace('{','')
                idx = idx + 1
                print(f"\n\n{idx}. {title} abstract:\n")
                passage += f"\n\n{idx}. {title} abstract:\n"
                passage += f"<C{i}>"+c['content']
                
        
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
            ground_truth = js['paper']
            # del js['few_shot_scores']
            prediction = re.sub(r"<cite>.*?</cite>", "", current_output, flags=re.DOTALL)
            prediction = prediction.replace('<statement>', '').replace('</statement>', '')
            gpt_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'responses': []}
            correct_score = 0
            response = dataset2metric[dataset](title,abstract,prediction, ground_truth, query=query, gpt_usage=gpt_usage)
            score = response['scores']
            evaluation = response['evaluation']
            total_score = score['total']
            mean_score = int(total_score)/25
            
            correct_score = max(correct_score,mean_score)
            detailed_score ={
                'relevance': score['relevance'],
                'coherence': score['coherence'],
                'academic': score['academic'],
                'completeness': score['completeness'],
                'innovation': score['innovation'],
                'total': total_score,
                'mean_score': mean_score
            }
            
            js['correct_score'] = correct_score
            js['gpt_usage'] = gpt_usage
            res = {
                'idx': js['idx'],
                'dataset': js['dataset'],
                'abstract': js['abstract'],
                'ground_truth': js['paper'],
                'prediction': current_output,
                'statements': statements_with_citations,
                'evaluation': {
                    'citation_recall': cur_evaluation_result['citation_recall'],
                    'citation_precision': cur_evaluation_result['citation_precision'],
                    'citation_f1': cur_evaluation_result['citation_f1'],
                    'correct_score': correct_score,
                    'detailed_score': detailed_score,
                    'detailed_interpretaion_for_correctness': evaluation,
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
            comprehensive_score = correct_score * 0.3 + cur_evaluation_result['citation_recall'] * 0.7
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
                'abstract': js['abstract'],
                'ground_truth': js['paper'],
                'prediction': current_output,
                'statements': statements_with_citations,
                'evaluation': {
                    'citation_recall': cur_evaluation_result['citation_recall'],
                    'citation_precision': cur_evaluation_result['citation_precision'],
                    'citation_f1': cur_evaluation_result['citation_f1'],
                    'correct_score': correct_score,
                    'detailed_score': detailed_score,
                    'detailed_interpretaion_for_correctness':evaluation,
                    'gpt_usage_for_citation_eval': cur_evaluation_result['gpt_usage'],
                    'gpt_usage_for_correctness_eval': gpt_usage
                },
                'iterations': iterations_results
            }
        
        best_result = max(iterations_results,key=lambda x:x['comprehensive_score'])
        res['best_result'] = best_result
        
        with jsonlines.open(fout_path,mode='a') as writer:
            writer.write(res)
        
        return res
    except:
        print(js['idx'])
        print(query)
        traceback.print_exc()
        print('-'*100)
        return None

def process_with_iterations_v2_enhanced(js, max_iterations=3, score_threshold=0.7):
    """
    增强版的process_with_iterations_v2函数，包含评估和迭代优化功能
    """
    
    generator = Generator(
        llm_provider="gemini",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        model_name="google/gemini-2.5-flash-preview-05-20",
        database_uri=os.environ.get("MILVUS_URI"),
        database_token=os.environ.get("MILVUS_TOKEN"),
    )
    
    # 初始化评估客户端
    from citegeist.utils.azure_client import AzureClient
    from citegeist.utils.helpers import load_api_key
    from citegeist.utils.prompts import (
        generate_relevance_evaluation_prompt,
        generate_win_rate_evaluation_prompt
    )
    from citegeist.utils.citations import get_arxiv_abstract
    
    evaluation_client = AzureClient(
        endpoint=os.getenv("AZURE_ENDPOINT"),
        deployment_id=os.getenv("AZURE_PROMPTING_MODEL"),
        api_key=load_api_key(os.getenv("KEY_LOCATION")),
    )
    
    # 参数设置
    abstract = js['abstract']
    breadth = 10
    depth = 2
    diversity = 0.0
    
    # 存储迭代结果
    iterations_results = []
    best_result = None
    best_score = 0.0
    
    for iteration in range(max_iterations):
        print(f"\n=== 迭代 {iteration + 1}/{max_iterations} ===")
        
        # 根据迭代次数调整参数
        if iteration > 0:
            # 增加多样性以探索更多可能性
            diversity = min(0.3, diversity + 0.1)
            # 增加深度以获取更多细节
            depth = min(3, depth + 1)
        
        # 生成相关工作
        result = generator.generate_related_work(abstract, breadth, depth, diversity)
        
        # ========== 评估环节 ==========
        evaluation_result = evaluate_related_work(
            result, js, abstract, evaluation_client
        )
        
        # 计算综合评分
        comprehensive_score = calculate_comprehensive_score(evaluation_result)
        
        # 记录本次迭代结果
        iteration_result = {
            'iteration': iteration + 1,
            'result': result,
            'evaluation': evaluation_result,
            'comprehensive_score': comprehensive_score,
            'parameters': {
                'breadth': breadth,
                'depth': depth,
                'diversity': diversity
            }
        }
        iterations_results.append(iteration_result)
        
        # 更新最佳结果
        if comprehensive_score > best_score:
            best_score = comprehensive_score
            best_result = iteration_result
        
        # 打印评估结果
        print(f"迭代 {iteration + 1} 评估结果:")
        print(f"- 与原始工作对比: {evaluation_result['original_vs_generated']}")
        print(f"- 与GPT工作对比: {evaluation_result['generated_vs_gpt']}")
        print(f"- 引用数量: {evaluation_result['citation_count']}")
        print(f"- 综合评分: {comprehensive_score:.3f}")
        
        # 检查是否达到目标分数
        if comprehensive_score >= score_threshold:
            print(f"达到目标分数 {score_threshold}，停止迭代")
            break
        
        # 检查是否有改进
        if iteration > 0 and comprehensive_score <= iterations_results[-2]['comprehensive_score']:
            print(f"评分没有改进，停止迭代")
            break
    
    # 返回最佳结果
    final_result = best_result['result'] if best_result else result
    final_result['evaluation'] = best_result['evaluation'] if best_result else evaluation_result
    final_result['iterations'] = iterations_results
    final_result['best_score'] = best_score
    
    return final_result


def evaluate_related_work(result, js, abstract, evaluation_client):
    """
    评估生成的相关工作质量
    """
    evaluation_result = {}
    
    # 1. 与原始相关工作对比
    if 'related_works' in js and js['related_works']:
        win_rate_prompt, order = generate_win_rate_evaluation_prompt(
            source_abstract=abstract,
            source_related_work=js['related_works'],
            target_related_work=result['related_works']
        )
        
        win_rate_response = evaluation_client.get_completions(
            win_rate_prompt, 
            os.getenv("AZURE_PROMPTING_MODEL_VERSION")
        )
        
        selected_winner_original_vs_generated = ""
        if win_rate_response == "Section A":
            selected_winner_original_vs_generated = order[0]
        elif win_rate_response == "Section B":
            selected_winner_original_vs_generated = order[1]
        elif win_rate_response == "Tie":
            selected_winner_original_vs_generated = "tie"
    else:
        selected_winner_original_vs_generated = "no_original_work"
    
    # 2. 与GPT生成相关工作对比
    if 'related_works_gpt4o_mini' in js and js['related_works_gpt4o_mini']:
        win_rate_prompt2, order2 = generate_win_rate_evaluation_prompt(
            source_abstract=abstract,
            source_related_work=result['related_works'],
            target_related_work=js['related_works_gpt4o_mini']
        )
        
        win_rate_response2 = evaluation_client.get_completions(
            win_rate_prompt2,
            os.getenv("AZURE_PROMPTING_MODEL_VERSION")
        )
        
        selected_winner_generated_vs_gpt = ""
        if win_rate_response2 == "Section A":
            selected_winner_generated_vs_gpt = order2[0]
        elif win_rate_response2 == "Section B":
            selected_winner_generated_vs_gpt = order2[1]
        elif win_rate_response2 == "Tie":
            selected_winner_generated_vs_gpt = "tie"
    else:
        selected_winner_generated_vs_gpt = "no_gpt_work"
    
    # 3. 引用质量评估
    citation_count = len(result['citations']) if result['citations'] else 0
    
    # 4. 文本质量评估（简单启发式）
    text_quality_score = evaluate_text_quality(result['related_works'])
    
    evaluation_result = {
        'original_vs_generated': selected_winner_original_vs_generated,
        'generated_vs_gpt': selected_winner_generated_vs_gpt,
        'citation_count': citation_count,
        'text_quality_score': text_quality_score,
        'win_rate_prompt_original': win_rate_prompt if 'related_works' in js and js['related_works'] else None,
        'win_rate_prompt_gpt': win_rate_prompt2 if 'related_works_gpt4o_mini' in js and js['related_works_gpt4o_mini'] else None,
        'win_rate_response_original': win_rate_response if 'related_works' in js and js['related_works'] else None,
        'win_rate_response_gpt': win_rate_response2 if 'related_works_gpt4o_mini' in js and js['related_works_gpt4o_mini'] else None
    }
    
    return evaluation_result


def calculate_comprehensive_score(evaluation_result):
    """
    计算综合评分
    """
    score = 0.0
    
    # 与原始工作对比评分
    if evaluation_result['original_vs_generated'] == "generated":
        score += 0.3
    elif evaluation_result['original_vs_generated'] == "tie":
        score += 0.15
    elif evaluation_result['original_vs_generated'] == "original":
        score += 0.05
    
    # 与GPT工作对比评分
    if evaluation_result['generated_vs_gpt'] == "generated":
        score += 0.3
    elif evaluation_result['generated_vs_gpt'] == "tie":
        score += 0.15
    elif evaluation_result['generated_vs_gpt'] == "gpt":
        score += 0.05
    
    # 引用数量评分
    citation_count = evaluation_result['citation_count']
    if 3 <= citation_count <= 8:
        score += 0.2
    elif citation_count > 8:
        score += 0.1
    elif citation_count > 0:
        score += 0.05
    
    # 文本质量评分
    score += evaluation_result['text_quality_score'] * 0.2
    
    return min(1.0, score)


def evaluate_text_quality(text):
    """
    简单的文本质量评估
    """
    score = 0.0
    
    # 检查长度
    if len(text) > 500:
        score += 0.2
    
    # 检查段落数量
    paragraphs = text.split('\n\n')
    if 2 <= len(paragraphs) <= 5:
        score += 0.2
    
    # 检查是否包含引用
    if '(' in text and ')' in text:
        score += 0.2
    
    # 检查学术词汇
    academic_words = ['research', 'study', 'analysis', 'method', 'approach', 'technique', 'framework', 'model']
    academic_word_count = sum(1 for word in academic_words if word.lower() in text.lower())
    if academic_word_count >= 3:
        score += 0.2
    
    # 检查句子结构
    sentences = text.split('.')
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
    if 10 <= avg_sentence_length <= 25:
        score += 0.2
    
    return min(1.0, score)

# ========== 使用示例和说明 ==========

def example_usage():
    """
    使用示例：如何在process_with_iterations_v2中加入评估环节
    """
    
    # 示例输入数据
    sample_data = {
        'abstract': 'This paper presents a novel approach to machine learning...',
        'related_works': 'Previous work in this area includes...',  # 原始相关工作
        'related_works_gpt4o_mini': 'Recent advances in machine learning...',  # GPT生成的相关工作
        'idx': 1
    }
    
    print("=== 基础版本（单次生成 + 评估）===")
    result_basic = process_with_iterations_v2(sample_data)
    print(f"生成的相关工作: {result_basic['related_works'][:200]}...")
    print(f"评估结果: {result_basic['evaluation']}")
    
    print("\n=== 增强版本（迭代优化 + 评估）===")
    result_enhanced = process_with_iterations_v2_enhanced(
        sample_data, 
        max_iterations=3, 
        score_threshold=0.7
    )
    print(f"最佳评分: {result_enhanced['best_score']:.3f}")
    print(f"迭代次数: {len(result_enhanced['iterations'])}")


def evaluation_guidance():
    """
    评估环节的设计指导
    """
    print("""
=== 评估环节设计指导 ===

1. 借鉴evaluation-optimized-pipeline的核心思想：
   - 使用LLM作为评判者进行对比评估
   - 采用win-rate评估方法
   - 多维度评估（相关性、质量、引用等）

2. 评估维度：
   - 与原始相关工作对比（人工编写）
   - 与GPT生成相关工作对比（基线方法）
   - 引用质量和数量
   - 文本结构和学术性

3. 评分机制：
   - 综合评分 = 对比评估得分 + 引用质量得分 + 文本质量得分
   - 权重可调整：对比评估60%，引用质量20%，文本质量20%

4. 迭代优化策略：
   - 动态调整参数（diversity, depth）
   - 基于评分结果选择最佳版本
   - 设置停止条件（达到阈值或无改进）

5. 环境变量配置：
   - AZURE_ENDPOINT: Azure OpenAI端点
   - AZURE_PROMPTING_MODEL: 用于评估的模型
   - KEY_LOCATION: API密钥位置
   - AZURE_PROMPTING_MODEL_VERSION: 模型版本

6. 返回结果结构：
   {
       'related_works': '生成的相关工作文本',
       'citations': ['引用列表'],
       'evaluation': {
           'original_vs_generated': '对比结果',
           'generated_vs_gpt': '对比结果',
           'citation_count': 引用数量,
           'evaluation_score': 综合评分
       },
       'iterations': [迭代历史],
       'best_score': 最佳评分
   }
""")


if __name__ == "__main__":
    # 调试模式
    DEBUG = True
    if DEBUG:
        # for item in need_list[:]:
        #     print(f"\n处理样本 {item['idx']}:")
        #     print(f"问题: {item['abstract']}")
        #     result = process_with_iterations_v2(item)
        #     print(f"Related Work:\n{result['related_works']}")
        #     print(f"Citations:\n{result['citations']}")
        # 单进程调试模式
        for item in need_list[:]:  # 处理所有样本
            print(f"\n处理样本 {item['idx']}:")
            print(f"问题: {item['abstract']}")
            result = process_with_iterations_v2(item)
            # process_with_iterations(item)
            # if result:
            #     print("\n最终结果:")
            #     print(f"最佳迭代次数: {result['best_result']['iteration']}")
            #     print(f"最终分数: {result['best_result']['comprehensive_score']:.3f}")
            #     print(f"最终正确性: {result['best_result']['correct_score']:.3f}")
            #     print(f"最终召回: {result['best_result']['evaluation']['citation_recall']:.3f}")
            #     print(f"最终精确: {result['best_result']['evaluation']['citation_precision']:.3f}")
            #     print(f"最终F1: {result['best_result']['evaluation']['citation_f1']:.3f}")
            #     print("")
            #     print("\n总体评估结果:")
            #     print(f"Recall: {result['evaluation']['citation_recall']:.3f}")
            #     print(f"Precision: {result['evaluation']['citation_precision']:.3f}")
            #     print(f"F1: {result['evaluation']['citation_f1']:.3f}")
            #     print(f"正确性得分: {result['evaluation']['correct_score']:.3f}")
    else:
        # 正常多进程模式
        with Pool(4) as p:
            p.map(process,need_list)