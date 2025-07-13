import os, sys
import json
import re
import numpy as np
sys.path.append('../')
from llm_api import query_llm

GPT_MODEL = "deepseek-chat"
need_citation_prompt_template = """You are an expert in academic writing and citation verification. You will receive:

A researcher's abstract about an academic paper,

An AI assistant's generated introduction based on the abstract,

A citation statement from the introduction,


Your task is to determine whether this sentence is a factual statement made based on the information in the document that requires citation, 
rather than an introductory sentence, transition sentence, or a summary, reasoning, or inference based on the previous response.

Ensure that you do not use any other external information during your evaluation. 
Please first provide your judgment (answer with [[Yes]] or [[No]]), then provide your analysis in the format "Need Citation: [[Yes/No]] Analysis: ...".\n\n{}
"""

def cat_qa_and_statement(question, answer, statement):
    prompt = f"<abstract>\n{question.strip()}\n</abstract>\n\n<introduction>\n{answer.strip()}\n</introduction>\n\n<sentence>\n{statement.strip()}\n</sentence>"
    return prompt

def need_citation_to_score(s):
    l = re.findall(r'\[\[([ /a-zA-Z]+)\]\]', s)
    if l:
        l = l[0]
        if "yes".lower() in l.lower():
            return 1
        else:
            return 0
    else:
        return None

def need_citation(question, answer, sentence):
    prompt = need_citation_prompt_template.format(cat_qa_and_statement(question, answer, sentence))
    for t in range(5):
        msg = [{'role': 'user', 'content': prompt}]
        output = query_llm(msg, model=GPT_MODEL, temperature=0 if t == 0 else 1, max_new_tokens=10, stop="Analysis:", return_usage=True)
        if isinstance(output, tuple):
            output, usage = output
            score = need_citation_to_score(output)
            usage = usage
            print(usage)
        else:
            score, usage = None, None
        if score is None:
            print("Unexcept need_citation output: ", output)
            if 'Trigger' in output:
                break
            continue
        else:
            break
    return score, output, usage
    
support_prompt_template = '''You are an expert in academic writing and citation verification. You will receive:

A researcher's abstract about an academic paper,

A citation statement from an AI assistant's generated introduction based on the abstract,

A relevant excerpt from a cited paper

Evaluate whether the citation is substantiated by the excerpt using this scale:

[[Fully supported]] - The statement accurately reflects the excerpt's claims, including key findings, methodologies or conclusions. Applies only when claims are directly verifiable.

[[Partially supported]] - The statement captures some elements correctly but misrepresents/misses other significant aspects (e.g. overgeneralizing findings, omitting limitations).

[[No support]] - The claimed citation contradicts or bears no meaningful connection to the excerpt.

Important:

Consider only explicit evidence in the excerpt

Flag any statistical claims without specific values

Note when claims exceed the excerpt's scope

Verify proper contextualization of quoted material

Please provide the rating first, followed by the analysis, in the format "Rating: [[...]] Analysis: ...". \n\n{}

'''



def cat_question_statement_context(question, statement, context):
    prompt = f"<abstract>\n{question.strip()}\n</abstract>\n\n<statement>\n{statement.strip()}\n</statement>\n\n<snippet>\n{context.strip()}\n</snippet>\n\n"
    return prompt

def support_level_to_score(s):
    l = re.findall(r'\[\[([ /a-zA-Z]+)\]\]', s)
    if l:
        l = l[0]
        if "fully".lower() in l.lower():
            return 1
        elif "partially".lower() in l.lower():
            return 0.5
        else:
            return 0
    else:
        return None
    
def is_support(question, statement, context):
    if context == "":
        return 0, "No matched citation", None
    prompt = support_prompt_template.format(cat_question_statement_context(question, statement, context))
    for t in range(5):
        msg = [{'role': 'user', 'content': prompt}]
        output = query_llm(msg, model=GPT_MODEL, temperature=0 if t == 0 else 1, max_new_tokens=10, stop="Analysis:", return_usage=True)
        if isinstance(output, tuple):
            output, usage = output
            score = support_level_to_score(output)
        else:
            score, usage = None, None
        if score is None:
            print("Unexcept support output: ", output)
            if 'Trigger' in output:
                break
            continue
        else:
            break
    return score, output, usage
    
def score_recall(question, answer, statements_with_citations):
    scores, usages = [], []
    for js in statements_with_citations:
        statement, citations = js['statement'], js['citation']
        matched_citations = [c['cite'] for c in citations]
        if len(matched_citations) > 0:
            context = '\n\n'.join(matched_citations).strip()
            score, output, usage = is_support(question, statement, context)
            usages.append(usage)
            js.update({
                "support_output": output,
                "support_score": score,
            })
            if score is None:
                print("ERROR\tUnexcept support output: ", statement + '\t>>\t' + output)
                raise NotImplementedError
            else:
                scores.append(score)
        else:
            score, output, usage = need_citation(question, answer, statement)
            usages.append(usage)
            js.update({
                "support_output": output,
                "support_score": 1 - score if score is not None else None,
            })
            if score is None:
                print("ERROR\tUnexcept need_citation output: ", output)
                raise NotImplementedError
            else:
                scores.append(1-score)
    if len(scores) == 0:
        return 0, statements_with_citations, usages
    else:
        return np.mean(scores), statements_with_citations, usages
        
relevant_prompt_template = """You are an expert in academic writing and citation verification. You will receive:

A researcher's abstract about an academic paper,

A citation statement from an AI assistant's response claiming to reference that paper,

A relevant excerpt from the cited paper (as full text cannot be displayed).

Your task is to carefully assess whether the snippet contains some key information of the statement. Please use the following grades to generate the rating:
- [[Relevant]] - Some key points of the statement are supported by the snippet or extracted from it.
- [[Unrelevant]] - The statement is almostly unrelated to the snippet, or all key points of the statement are inconsistent with the snippet content.
Ensure that you do not use any information or knowledge outside of the snippet when evaluating. 
Please provide the rating first, followed by the analysis, in the format "Rating: [[...]] Analysis: ...". \n\n{}"""

def relevant_level_to_score(s):
        l = re.findall(r'\[\[([ /a-zA-Z]+)\]\]', s)
        if l:
            l = l[0]
            if "unrelevant".lower() in l.lower():
                return 0
            else:
                return 1
        else:
            return None
        
def is_relevant(question, statement, citation):
    prompt = relevant_prompt_template.format(cat_question_statement_context(question, statement, citation))
    for t in range(5):
        msg = [{'role': 'user', 'content': prompt}]
        output = query_llm(msg, model=GPT_MODEL, temperature=0 if t == 0 else 1, max_new_tokens=256, stop="Analysis:", return_usage=True)

        if isinstance(output, tuple):
            output, usage = output
            score = relevant_level_to_score(output)
        else:
            score = relevant_level_to_score(output)
            usage = None
        if score is None:
            print("Unexcept relevant output: ", output)
            if 'Trigger' in output:
                break
            continue
        else:
            break
    return score, output, usage

def score_precision(question, answer, statements_with_citations):
    scores, usages = [], []
    for js in statements_with_citations:
        statement, citations = js['statement'], js['citation']
        for c in citations:
            score, output, usage = is_relevant(question, statement, c['cite'])

            usage = usage
            usages.append(usage)
            c.update({
                "relevant_output": output,
                "relevant_score": score,
            })
            if score is None:
                print("ERROR\tUnexcept relevant output: ", output)
                raise NotImplementedError
            else:
                scores.append(score)
        
    if len(scores) == 0:
        return 0, statements_with_citations, usages
    else:
        return np.mean(scores), statements_with_citations, usages

def get_citation_score(js, max_statement_num=None, real_time=False):
    question, answer, statements_with_citations = js['query'], js['prediction'], js['statements']
    answer = re.sub(r"<cite>.*?</cite>", "", answer, flags=re.DOTALL)
    answer = answer.replace('<statement>', '').replace('</statement>', '')
    
    if max_statement_num and len(statements_with_citations) > max_statement_num:
        print(f"Too many statments, only evaluate {max_statement_num} of {len(statements_with_citations)} statements.")
        statements_with_citations = statements_with_citations[:max_statement_num]
    
    # 实时评估模式
    if real_time:
        current_scores = []
        current_usages = []
        
        for i, statement in enumerate(statements_with_citations):
            # 评估当前语句
            recall_score, _, usage1 = score_recall(question, answer, [statement])
            precision_score, _, usage2 = score_precision(question, answer, [statement])
            
            current_scores.append({
                'recall': recall_score,
                'precision': precision_score,
                'f1': 0.0 if recall_score + precision_score == 0 else (2 * recall_score * precision_score) / (recall_score + precision_score)
            })
            current_usages.extend([usage1, usage2])
            
            # 打印实时评估结果
            print(f"\n语句 {i+1} 评估结果:")
            print(f"Recall: {recall_score:.3f}")
            print(f"Precision: {precision_score:.3f}")
            print(f"F1: {current_scores[-1]['f1']:.3f}")
        
        # 计算总体分数
        avg_recall = np.mean([s['recall'] for s in current_scores])
        avg_precision = np.mean([s['precision'] for s in current_scores])
        avg_f1 = np.mean([s['f1'] for s in current_scores])
        
        js['citation_recall'] = avg_recall
        js['citation_precision'] = avg_precision
        js['citation_f1'] = avg_f1
        js['gpt_usage'] = {
            'prompt_tokens': sum(x['prompt_tokens'] for x in current_usages),
            'completion_tokens': sum(x['completion_tokens'] for x in current_usages),
        }
        js['real_time_scores'] = current_scores
        
    else:
        # 原有的批量评估逻辑
        recall, _, usages1 = score_recall(question, answer, statements_with_citations)
        precision, _, usages2 = score_precision(question, answer, statements_with_citations)
        js['citation_recall'] = recall
        js['citation_precision'] = precision
        js['citation_f1'] = 0.0 if recall + precision == 0 else (2 * recall * precision) / (recall + precision)
        js['gpt_usage'] = {
            'prompt_tokens': sum(x['prompt_tokens'] for x in (usages1 + usages2)),
            'completion_tokens': sum(x['completion_tokens'] for x in (usages1 + usages2)),
        }
    
    return js