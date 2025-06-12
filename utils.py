from nltk.tokenize import PunktSentenceTokenizer
from copy import deepcopy
import re

def get_citations(statement, sents):
    c_texts = re.findall(r'<cite>(.*?)</cite>', statement, re.DOTALL)
    c_texts = [c_text.replace('C', '') if 'C' in c_text else c_text for c_text in c_texts]
    # c_texts = []
    
    # for c_text in c_texts:
    #     if 'C' in c_text:
    #         c_text = c_text.replace('C', '')
    #         c_texts.append(c_text)
    #     else:
    #         c_texts.append(c_text)
    #  c_texts = [c_text.replace('C', '') for c_text in c_texts if 'C' in c_text]
    spans = sum([re.findall(r"\[([0-9]+\-[0-9]+)\]", c_text, re.DOTALL) for c_text in c_texts], [])
    statement = re.sub(r'<cite>(.*?)</cite>', '', statement, flags=re.DOTALL)
    merged_citations = []
    for i, s in enumerate(spans):
        try:
            st, ed = [int(x) for x in s.split('-')]
            if st > len(sents) - 1 or ed < st:
                continue
            st, ed = max(0, st), min(ed, len(sents)-1)
            assert st <= ed, str(c_texts) + '\t' + str(len(sents))
            if len(merged_citations) > 0 and st == merged_citations[-1]['ed_sent'] + 1:
                merged_citations[-1].update({
                    "ed_sent": ed,
                    'end_char': sents[ed]['end'],
                    'cite': ''.join([x['content'] for x in sents[merged_citations[-1]['st_sent']:ed+1]]),
                })
            else:
                merged_citations.append({
                    "st_sent": st,
                    "ed_sent": ed,
                    "start_char":  sents[st]['start_idx'],
                    'end_char': sents[ed]['end_idx'],
                    'cite': ''.join([x['content'] for x in sents[st:ed+1]]),
                })
        except:
            print(c_texts, len(sents), statement)
            raise
    return statement, merged_citations[:3]

def text_split_by_punctuation(original_text,return_dict=False):
    """
    将原始文本按标点符号分割成多个句子
    """
    text = original_text
    # text = text.replace("“","").replace("”","")
    custom_sent_tokenizer = PunktSentenceTokenizer()
    punctuations = r"([。；！？])"
    
    separated = custom_sent_tokenizer.tokenize(text)
    separated = sum([re.split(punctuations,sentence) for sentence in separated],[])
    
    for i in range(1,len(separated)):
        if re.match(punctuations,separated[i]):
            separated[i-1] += separated[i]
    separated = [s.strip() for s in separated if s.strip() != ""]
    
    if not return_dict:
        return separated
    else:
        pos = 0
        res = []
        for i, sent in enumerate(separated):
            if sent in ["。","？","！","；"]:
                continue
            st = original_text.find(sent, pos)
            assert st != -1, sent
            ed = st + len(sent)
            res.append(
                {
                    'c_idx': i,
                    'content': sent,
                    'start_idx': st,
                    'end_idx': ed,
                }
            )
            pos = ed
        return res
    
def postprocess(answer,context,sents):
    # 将context按句子分割成多个句子
    chunks = []
    for k,s in enumerate(sents):
        s_start,s_end = s['start_idx'],s['end_idx']
        assert s['content'] == context[s_start:s_end],s
        s_end = sents[k+1]['start_idx'] if k < len(sents) - 1 else s['end_idx']
        chunks.append({
            'content':context[s_start:s_end],
            'start_idx':s_start,
            'end_idx':s_end,
            'c_idx':s['c_idx']
        } 
    )
    
    res = []
    pos = 0
    while True:
        start = answer.find("<statement>",pos)
        if start == -1:
            start = len(answer)
        end = answer.find("</statement>",start)
        statement = answer[pos:start]
        if len(statement.strip()) > 5:
            res.append(
                {
                    "statement":statement,
                    "citation":[]
                }
            )
        if end == -1:
            break
        
        statement = answer[start+len("<statement>"):end]
        
        if len(statement.strip()) > 0:
            statement,citations = get_citations(statement,sents)
            res.append(
                {
                    "statement":statement,
                    "citation":citations
                }
            )
        pos = end + len("</statement>")
        
    return res


