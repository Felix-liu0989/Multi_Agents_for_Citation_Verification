from llm_api import query_llm


with open('D:\Mydesktop\CitAgent\Code\LongCite-main\LongBench-Cite\one_shot_prompt.txt', "r",encoding='utf-8') as fp:
    prompt_format = fp.read()

def generate_feedback(query, prediction, evaluation_result,cite_model):
    feedback_prompt = f"""As an evaluation expert, please analyze and provide feedback on the following response:

Question: {query}

Current Response: {prediction}

Evaluation results:
- Citation precision: {evaluation_result['evaluation']['citation_precision']:.3f}
- Citation recall: {evaluation_result['evaluation']['citation_recall']:.3f}
- Citation F1 score: {evaluation_result['evaluation']['citation_f1']:.3f}
- Answer correctness: {evaluation_result['evaluation']['detailed_score']['mean_score']:.3f}

## Feedback Instruction

- Correctness: 
If the reward score is below 0.7, 
provide feedback to generate more relevant responses based on the search result summaries. 
If the score is above 0.7, affirm that performance is satisfactory.

- Citation Recall: 
If the reward score is below 0.7, 
provide feedback to offer citations from credible sources for each factual statement you make. 
If the score is above 0.7, affirm that performance on citation recall is satisfactory.

- Citation Precision: 
If the reward score is below 0.7, provide feedback to cite properly, 
ensuring all factual statements refer to an appropriate search result. 
If the score is above 0.7, affirm that performance on citation precision is satisfactory.

Please provide specific feedback in the following aspects:
1. Whether citations are accurate and sufficient
2. Whether the answer is correct and complete
3. Specific areas needing improvement

Feedback:"""

    msg = [{'role': 'system', 'content': feedback_prompt}]
    feedback = query_llm(
        messages=msg,
        model=cite_model,
        temperature=0.7,
        max_new_tokens=512,
        return_usage=False
    )
    return feedback

def optimize_response(query, prediction, feedback, cite_model,context):
    """Optimize response based on feedback"""
    base_prompt = prompt_format.replace('<<context>>',context).replace('<<question>>',query)
    optimize_prompt = f"""
{base_prompt}
Original response: {prediction}
Based on the following feedback, please optimize the original response:

Feedback: {feedback}

Please generate an improved response ensuring:
1. Fix citation issues
2. Improve answer accuracy
3. Maintain response coherence

Optimized response:"""

    msg = [{'role': 'system', 'content': optimize_prompt}]
    optimized_response = query_llm(
        messages=msg,
        model=cite_model,
        temperature=0.7,
        max_new_tokens=1024,
        return_usage=False
    )
    return optimized_response