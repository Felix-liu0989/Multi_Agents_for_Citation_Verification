# flake8: noqa
def process_data_for_related_work_prompt(related_work:str) -> str:
    """
    
    """
    return f"""
    Below is the paper's sections including related work:
    {related_work}
    
    Please review the related work and extract the subsections of the related work if any.
    Please exclusively respond with the subsections. Do not add any filler text before or after the subsections.
    Also, do not use any type of markdown formatting. 
    
    **Output Format:**
    if there is related work without subsections, return a whole related work content without any formatting,only return the content.I want a pure text output only.
    if there is related work with subsections, return a json object with the following structure:
    {{
        "subsection1": "subsection1 content",
        "subsection2": "subsection2 content",
        ...
    }}
    """


def genrate_original_related_work_feedback_prompt(
    related_work: str,
) -> str:
    """
    Generate the prompt for original related work feedback.
    """
    return f"""
    Below is the original related work:
    "{related_work}"
    
    Please provide feedback on the original related work.
    - structural logic, 
    - critical analysis, 
    - prominent contribution, 
    - concise language.
    
    Please exclusively respond with the feedback. 
    Do not add any filler text before or after the feedback. 
    Also, do not use any type of markdown formatting. 
    I want a pure text output only.
    """

def generate_related_work_comparative_prompt(
    source_abstract: str,
    related_work_summary: str,
    current_reference_abstract: str,
    current_reference_title: str,
    current_reference_citation: str,
) -> str:
    """
    Generate the prompt for comparative analysis of related work.
    """
    return f"""
    Below is the abstract of my paper:
    "{source_abstract}"
    
    You are an expert academic writer tasked with generating a comparative literature summary. 
    Your goal is to compare the current reference paper with my paper, while maintaining coherence with the existing summary draft.
    
    Below is the title, abstract and citation of the current reference paper:
    Title: "{current_reference_title}"
    Abstract: "{current_reference_abstract}"
    Citation: "{current_reference_citation}"
    
    Below is the current summary draft of the related work (if any):
    "{related_work_summary}"
    
    Please exclusively respond with the comparative summary. 
    Do not add any filler text before or after the summary. 
    Also, do not use any type of markdown formatting. 
    I want a pure text output only.
    """



def generate_brief_topic_prompt(abstract:str) -> str:
    """
    生成凝练的topic，为生成相关文献的分类体系作准备
    """
    return f"""
    Below is the abstract of my paper:
    "{abstract}"
    Please generate a brief topic for my paper. The topic should be a single phrase （within 10 words） that captures the main idea of the paper.
    Please exclusively respond with the topic. Do not add any filler text before or after the topic.
    Also, do not use any type of markdown formatting. 
    I want a pure text output only.
    """

def generate_related_work_outline_prompt_various_1(source_abstract:str,arxiv_trees:str) -> str:
    """
    生成related work大纲
    """
    return f"""
    Below is the abstract of my paper:
    "{source_abstract}"
    
    Below is the tree structure of the related work:
    "{arxiv_trees}"
    
    Please generate a outline for the related work. The outline should be a list of 2-4 topics that capture the main ideas of the related work.
    Each topic should be a single phrase (within 10 words) that:
    - Can cover most of the related work
    - Has clear research boundaries and distinctions
    - Is suitable as a main subsection title in the Related Work section of an academic paper
    - The topics should be different from each other
    - Each topic should have moderate coverage scope—neither too broad nor too narrow
    
    Please exclusively respond with the outline. Do not add any filler text before or after the outline.
    Also, do not use any type of markdown formatting. 
    I want a json output only.
    The json should be like this:
    {{
        "outline": [
            "topic1",
            "topic2",
            "topic3",
            ...
        ]
    }}
    """

def generate_related_work_outline_prompt(source_abstract:str,arxiv_trees:str) -> str:
    """
    生成related work大纲
    """
    return f"""
    Below is the abstract of my paper:
    "{source_abstract}"
    Below is the tree structure of the related work:
    "{arxiv_trees}"
    Please generate a outline for the related work. The outline should be a list of 2 topics that capture the main ideas of the related work.
    Each topic should be a single phrase (within 10 words) that:
    - Can cover most of the related work
    - Has clear research boundaries and distinctions
    - Is suitable as a main subsection title in the Related Work section of an academic paper
    - The two topics should be different from each other
    Please exclusively respond with the outline. Do not add any filler text before or after the outline.
    Also, do not use any type of markdown formatting. 
    I want a json output only.
    The json should be like this:
    {{
        "outline": [
            "topic1",
            "topic2"
        ]
    }}
    """

def generate_related_work_outline_prompt_various(source_abstract:str, arxiv_trees:str, topic1:str, topic2:str) -> str:
    return f"""
    Below is the abstract of my paper:
    "{source_abstract}"

    Below is the tree structure of the related works:
    "{arxiv_trees}"

    Below are the initially generated topics for the related work section:
    Topic 1: {topic1}
    Topic 2: {topic2}

    Please analyze and optimize these topics based on the following criteria:

    1.  **Overlap Analysis**: If topics have significant conceptual overlap, merge them into a single, more comprehensive topic.
    2.  **Coverage Analysis**: If topics are too broad and each could cover multiple distinct research areas, split them into more specific sub-topics.
    3.  **Distinctiveness Analysis**: If topics are appropriately distinct and well-scoped, keep them as they are.
    4.  **Completeness Analysis**: Ensure the final topics can comprehensively cover the research landscape related to the source abstract.

    Optimization Rules:
    - Final output should contain **2-4 topics** maximum (no more than 4)
    - Each topic should be a concise phrase (**within 10 words**)
    - Topics should have clear research boundaries
    - Topics should be suitable as **subsection titles** in academic papers

    **Additional Requirement**:
    - If the initial topics already fully comply with all the above criteria and require no optimization, directly return the boolean value `True`
    - If the initial topics are not suitable as subsection titles, please re-organize them into a list of 2-4 topics that are suitable as subsection titles
    
    **Output Format**:
    Please respond in the following format:
    If no optimization is needed:
        True
    
    If optimization is needed:
        {{
            "action_taken": "merged/split/kept/reorganized",
            "reasoning": "brief explanation of the optimization decision",
            "optimized_outline": [
                "optimized_topic1",
                "optimized_topic2",
                "optimized_topic3_if_needed",
                ...
            ]
        }}
    """
    

def generate_summary_prompt(abstract_source_paper: str, abstract_to_be_cited: str) -> str:
    """
    Generates the summary prompt for a given pair of abstracts.
    :param abstract_source_paper: Abstract of source paper
    :param abstract_to_be_cited: Abstract of a related work
    :return: Prompt string
    """
    return f"""
    Below are two abstracts:
    My abstract:
    "{abstract_source_paper}"
    Abstract of the paper I want to cite:
    "{abstract_to_be_cited}"
    Based on the two abstracts, write a brief few-sentence (at most 5) summary of the cited paper in relation to my work. Emphasize how the cited paper relates to my research.

    Please exclusively respond with the summary. Do not add any filler text before or after the summary. Also, do not use any type of markdown formatting. I want a pure text output only.
    """
def generate_related_work_revision_prompt(
    source_abstract: str,
    related_work: str,
    feedback: str,
    dimensions: list[str],
) -> str:
    """
    Generate the prompt for revising the related work section.
    """
    prompt_section_1 = f"""
    Below is the abstract of my paper:
    "{source_abstract}"
    Below is the related work section:
    "{related_work}"
    
    The related work section is divided into two subsections:
    - {dimensions[0]}
    - {dimensions[1]}
    
    Below is the feedback on the original related work:
    "{feedback}"
    
    Please revise the related work section from four aspects within each subsection: 
    - structural logic, 
    - critical analysis, 
    - prominent contribution, 
    - concise language.
    - Reorganized structure: Follow the logic of "problem → method classification → limitations → my innovation".
    - Strengthen criticism: After each type of method, add its shortcomings and relate them to my work.
    - Simplified language: Merge citations, delete redundant descriptions, and highlight the core viewpoints.
    - Clarify comparison: Directly state why my method is better in conclusion. 
    
    Please exclusively respond with the prompt. 
    Do not add any filler text before or after the prompt. 
    Also, do not use any type of markdown formatting. 
    I want a pure text output only.
    """
    prompt_section_2 = """
    **Output Format:**
    You must return ONLY a valid JSON object with the following structure:
    
    {{
        "related_work": "Your complete related work section text here...",
        "cite_ids": [
            {{"citation_text": "Smith et al., 2023", "paper_id": "id1"}},
            {{"citation_text": "Jones & Brown, 2022", "paper_id": "id2"}},
            {{"citation_text": "Wang (2024)", "paper_id": "id3"}}
        ]
    }}
    
    **Important Notes for cite_ids:**
    - The "citation_text" should be the EXACT text you used in the related_work section (e.g., "Smith et al., 2023") and should be totally the same as the citation format in the related_work section.
    - The "paper_id" should correspond to the paper_id from the input data
    - **CRITICAL: Record citations in the EXACT ORDER they appear in your related_work text**
    - If you cite the same paper multiple times, record each occurrence separately in the order they appear
    - Include ALL citations that appear in your related_work text
    - Be precise with author names and years as they appear in your citations

    **Example of correct ordering:**    
    If your related_work mentions: "Recent work by Johnson et al. (2023) shows... Building on this, Smith & Brown (2022) propose... Johnson et al. (2023) further demonstrate..."
    Then cite_ids should be:
    [
        {"citation_text": "Johnson et al. (2023)", "paper_id": "id1"},
        {"citation_text": "Smith & Brown (2022)", "paper_id": "id2"}, 
        {"citation_text": "Johnson et al. (2023)", "paper_id": "id3"}
    ]
    
   
    Do not include any additional text, explanations, or markdown formatting outside of the JSON response.
    """
    return prompt_section_1 + prompt_section_2

def generate_related_work_prompt_with_arxiv_trees(
    source_abstract: str,
    dimensions: list[str],
    grouped: dict[str, list[dict]],
) -> str:
    """
    Generates the related work prompt for a given pair of abstracts and an arxiv tree.
    """
    prompt_section_1 = f"""
    Below is the abstract of my paper:
    "{source_abstract}"

    Below is the grouped related work:
    """
    
    for dim in dimensions:
        prompt_section_1 += f"""
        Topic: {dim}
        """
        for paper in grouped[dim]:
            prompt_section_1 += f"""
            Paper id: {paper['paper_id']}
            Title: {paper['title']}
            Abstract: {paper['abstract']}
            Summary: {paper['summary']}
            Citations: {paper['citations']}
            """
    prompt_section_2 = f"""
    Instructions:
    Using all the information given above, your goal is to write a cohesive and well-structured "Related Work" section. This section should be organized into **exactly {len(dimensions)} distinct subsections**, each with a clear heading.
    """
    for i,dim in enumerate(dimensions):
        prompt_section_2 += f"""
        **Subsection {i+1}: {dim}**
        * **Content Focus:** In this subsection, discuss the problem of hallucination in Large Language Models (LLMs) and the emergence of "attributed text generation" as a crucial solution. Cover existing methods and paradigms designed to enhance the factual accuracy and verifiability of generated content by providing supporting evidence. This includes source-document grounded generation, Retrieval-Augmented Generation (RAG), and strategies that integrate fact verification and correction (e.g., knowledge-enhanced, iterative refinement, attribution-aware correction). Highlight the strengths of these approaches while also pointing out their limitations, especially concerning complex reasoning, multi-step information integration, or dynamic error correction, to set the stage for your paper's contribution.
        * **Paper Grouping:** Group papers from the provided list that primarily address the general concepts of attributed text generation, source grounding, RAG, and fact verification/correction methodologies.
    """
    prompt_section_2 += f"""
    **General Guidelines for Both Subsections:**
    * **Connections and Contrasts:** Draw clear connections between the related papers and your research, highlighting both similarities and key differences.
    * **Grouping and Cohesion:** Group papers with similar topics or implications into the same paragraph to maintain cohesion. Each paragraph should be substantial (avoid 2-3 sentence paragraphs).
    * **Comprehensive Coverage:** Ensure all provided papers from the `grouped` list are utilized and discussed within one of the two subsections.
    * **Proper Citation:** When referring to content from specific papers, you **must** cite the respective paper properly (e.g., `(Author et al., Year)` or `(Author A & Author B, Year)`). Cite directly after your direct or indirect quotes, or immediately after mentioning a specific method or finding from a paper. Do not use numerical citations like `[x]`.

    **Tone:** Maintain an academic, objective, and analytical tone throughout the "Related Work" section.
    """
    prompt_section_3 = """
    **Output Format:**
    You must return ONLY a valid JSON object with the following structure:
    
    {{
        "related_work": "Your complete related work section text here...",
        "cite_ids": [
            {{"citation_text": "Smith et al., 2023", "paper_id": "id1"}},
            {{"citation_text": "Jones & Brown, 2022", "paper_id": "id2"}},
            {{"citation_text": "Wang (2024)", "paper_id": "id3"}}
        ]
    }}
    
    **Important Notes for cite_ids:**
    - The "citation_text" should be the EXACT text you used in the related_work section (e.g., "Smith et al., 2023") and should be totally the same as the citation format in the related_work section.
    - The "paper_id" should correspond to the paper_id from the input data
    - **CRITICAL: Record citations in the EXACT ORDER they appear in your related_work text**
    - If you cite the same paper multiple times, record each occurrence separately in the order they appear
    - Include ALL citations that appear in your related_work text
    - Be precise with author names and years as they appear in your citations

    **Example of correct ordering:**    
    If your related_work mentions: "Recent work by Johnson et al. (2023) shows... Building on this, Smith & Brown (2022) propose... Johnson et al. (2023) further demonstrate..."
    Then cite_ids should be:
    [
        {"citation_text": "Johnson et al. (2023)", "paper_id": "id1"},
        {"citation_text": "Smith & Brown (2022)", "paper_id": "id2"}, 
        {"citation_text": "Johnson et al. (2023)", "paper_id": "id3"}
    ]
    
   
    Do not include any additional text, explanations, or markdown formatting outside of the JSON response.
    """
    return prompt_section_1 + prompt_section_2 + prompt_section_3

def generate_summary_prompt_question_with_page_content(
    question: str, abstract_to_be_considered: str, page_text_to_be_cited: list[str]
) -> str:
    """
    Generates the summary prompt for a given pair of abstracts.
    :param abstract_source_paper: Abstract of source paper
    :param abstract_to_be_cited: Abstract of a related work
    :return: Prompt string
    """
    output = f"""
    Below is a question and the abstract of a paper which may contain relevant information:
    My question:
    "{question}"

    Abstract of the paper that may contain relevant information:
    "{abstract_to_be_considered}"

    Relevant content of {len(page_text_to_be_cited)} pages within the paper:
    """

    for i in range(len(page_text_to_be_cited)):
        text = page_text_to_be_cited[i]
        output += f"""
        Page {i + 1}:
        "{text}"
        """

    output += f"""

    Based on the question and the paper abstract, write a brief few-sentence summary of the abstract in relation to the question. If the abstract does not contain relevant information, please reply with 'No relevant Information'.

    Please exclusively respond with the summary. Do not add any filler text before or after the summary. Also, do not use any type of markdown formatting. I want a pure text output only.
    """

    return output


def generate_summary_prompt_with_page_content(
    abstract_source_paper: str, abstract_to_be_cited: str, page_text_to_be_cited: list[str], sentence_count: int = 8
) -> str:
    """
    Generates the summary prompt for a given pair of abstracts and a list of relevant pages.
    :param abstract_source_paper: Abstract of source paper
    :param abstract_to_be_cited: Abstract of a related work
    :param page_text_to_be_cited: List of page text(s) of the related work
    :return: Prompt string
    """
    output = f"""
    Below are two abstracts and some content from a page of a paper:
    My abstract:
    "{abstract_source_paper}"

    Abstract of the paper I want to cite:
    "{abstract_to_be_cited}"

    Relevant content of {len(page_text_to_be_cited)} pages within the paper I want to cite:
    """

    for i in range(len(page_text_to_be_cited)):
        text = page_text_to_be_cited[i]
        output += f"""
        Page {i + 1}:
        "{text}"
        """

    output += f"""
    Based on the two abstracts and the content from the page, write a brief few-sentence (at most {str(sentence_count)}) summary of the cited paper in relation to my work. Emphasize how the cited paper relates to my research.

    Please exclusively respond with the summary. Do not add any filler text before or after the summary. Also, do not use any type of markdown formatting. I want a pure text output only.
    """
    return output


def generate_question_answer_prompt(question: str, data: list[dict]):
    output = f"""
    I am wondering about a scientific question, and I need a well-written answer that is based on scientific papers. Below I'm providing you with my question and a list of summaries of potentially relevant papers I've identified.

    Here's the question:
    "{question}"

    Here's the list of summaries of the other related works I've found:
    """

    for i in range(len(data)):
        summary = data[i]["summary"]
        citation = data[i]["citation"]
        output += f"""
        Paper {i + 1}:
        Summary: {summary}
        Citation: {citation}
        """

    output += f"""

    Instructions:
    Using all the information given above, your goal is to write a cohesive and well-structured answer to the given question. 
    Draw connections between the related papers and my question. 
    If multiple related works have a common point/theme, make sure to group them and refer to them in the same paragraph. 
    When referring to content from specific papers you must also cite the respective paper properly (i.e. cite right after your direct/indirect quotes, do not use [x]).
    Group papers with similar topics or implications into the same paragraph. 
    """
    return output

def generate_related_work_iterations(
    source_abstract:str,
    data: list[dict],
    paragraph_count: int = 5,
    add_summary: bool = True,
) -> str:
    """
    Generates the related work iterations prompt for an abstract and a set of summaries & citation strings.
    :param source_abstract: Abstract of source paper
    :param data: List of objects that each contain a paper summary and the respective citation string
    :return: Prompt string
    """
    output = f"""
    I am working on a research paper, and I need a well-written "Related Work" section.
    Below I'm providing you with the abstract of my paper and a list of summaries of related works I've identified.
    """
    return output

def generate_related_work_prompt(
    source_abstract: str, data: list[dict], paragraph_count: int = 5, add_summary: bool = True
) -> str:
    """
    Generates the related work prompt for an abstract and a set of summaries & citation strings.
    :param source_abstract: Abstract of source paper
    :param data: List of objects that each contain a paper summary and the respective citation string
    :return: Prompt string
    """
    output = f"""
    I am working on a research paper, and I need a well-written "Related Work" section. Below I'm providing you with the abstract of my paper and a list of summaries of related works I've identified.

    Here's the abstract of my paper:
    "{source_abstract}"

    """

    for i in range(len(data)):
        summary = data[i]["summary"]
        citation = data[i]["citation"]
        output += f"""
        Paper {i + 1}:
        Summary: {summary}
        Citation: {citation}
        """

    output += f"""

    Instructions:
    Using all the information given above, your goal is to write a cohesive and well-structured "Related Work" section. 
    Draw connections between the related papers and my research and highlight similarities and differences. 
    If multiple related works have a common point/theme, make sure to group them and refer to them in the same paragraph. 
    Please ensure that your generated section employs all the papers from above.
    When referring to content from specific papers you must also cite the respective paper properly (i.e. cite right after your direct/indirect quotes, do not use [x]).
    Group papers with similar topics or implications into the same paragraph. Limit yourself to at most {str(paragraph_count)} paragraphs, which should not be too short (e.g. avoid 2/3-sentence paragraphs).
    """
    
    output += """
    **Output Format:**
    You must return ONLY a valid JSON object with the following structure:
    {{
        "related_work": "Your complete related work section text here...",
        "cite_ids": [
            {{"citation_text": "Smith et al., 2023", "paper_id": "paper_001"}},
            {{"citation_text": "Jones & Brown, 2022", "paper_id": "paper_002"}},
            {{"citation_text": "Wang et al.", "paper_id": "paper_003"}}
        ]
    }}
    
    **Important Notes for cite_ids:**
    - The "citation_text" should be the EXACT text you used in the related_work section (e.g., "Smith et al., 2023")
    - The "paper_id" should correspond to the paper_id from the input data
    - **CRITICAL: Record citations in the EXACT ORDER they appear in your related_work text**
    - If you cite the same paper multiple times, record each occurrence separately in the order they appear
    - Include ALL citations that appear in your related_work text
    - Be precise with author names and years as they appear in your citations

    **Example of correct ordering:**    
    If your related_work mentions: "Recent work by Johnson et al. (2023) shows... Building on this, Smith & Brown (2022) propose... Johnson et al. (2023) further demonstrate..."
    Then cite_ids should be:
    [
        {"citation_text": "Johnson et al. (2023)", "paper_id": "paper_001"},
        {"citation_text": "Smith & Brown (2022)", "paper_id": "paper_002"}, 
        {"citation_text": "Johnson et al. (2023)", "paper_id": "paper_001"}
    ]
    
   
    Do not include any additional text, explanations, or markdown formatting outside of the JSON response.
    """
    if add_summary:
        output += "Please also make sure to put my work into the overall context of the provided related works in a summarizing paragraph at the end."
    return output


def generate_related_work_analysis_prompt(source_abstract: str, data: list[object]) -> str:
    """
    Generates the related work analysis prompt for an abstract and a set of summaries & citation strings.
    :param source_abstract: Abstract of source paper
    :param data: List of objects that each contain a paper summary and the respective citation string
    :return: Prompt string
    """
    output = f"""
    I am working on a research paper, and I would like to get a sense of related work for a specific section of my paper. Below I'm providing you with the section whose background I am interested in and a list of summaries of related works I've identified.

    Here's the section of my paper:
    "{source_abstract}"

    Here's the list of summaries of the other related works I've found:
    """

    for i in range(len(data)):
        summary = data[i]["summary"]
        citation = data[i]["citation"]
        output += f"""
        Paper {i + 1}:
        Summary: {summary}
        Citation: {citation}
        """

    output += """

    Instructions:
    Using all the information given above, your goal is to write a cohesive and well-structured analysis of the related work. 
    Draw connections between the related papers and my research section and highlight similarities and differences. 
    Please also make sure to put my section into the overall context of the provided related works in a summarizing paragraph at the end. 
    If multiple related works have a common points or themes, make sure to group them and refer to them in the same paragraph. 
    When referring to content from specific papers you must also cite the respective paper properly (i.e. cite right after your direct/indirect quotes).
    """
    return output


def generate_relevance_evaluation_prompt(source_abstract: str, target_abstract: str) -> str:
    """
    Generates an evaluation prompt to utilize LLM as a judge to determine the relevance with regard to the source abstract
    :param source_abstract: Abstract of source paper
    :param target_abstract: Abstract of target paper
    :return: Prompt String
    """
    prompt = f"""
        You are given two paper abstracts: the first is the source paper abstract, and the second is a related work paper abstract. Your task is to assess the relevance of the related work abstract to the source paper abstract on a scale of 0 to 10, where:

        - 0 means no relevance at all (completely unrelated).
        - 10 means the highest relevance (directly related and closely aligned with the source paper's topic and content).

        Consider factors such as:
        - Topic alignment: Does the related work paper address a similar research problem or area as the source paper?
        - Methodology: Does the related work discuss methods or techniques similar to those in the source paper?
        - Findings or contributions: Are the findings or contributions of the related work closely related to the source paper's content or conclusions?
        - The relationship between the two papers, such as whether the related work builds on, contrasts, or expands the source paper's work.

        Provide a score (0–10) and a brief explanation of your reasoning for the assigned score.

        Source Paper Abstract:
        {source_abstract}

        Related Work Paper Abstract:
        {target_abstract}

        Please provide only the score as your reply. Do not produce any other output, including things like formatting or markdown. Only the score.
    """
    return prompt


def generate_win_rate_evaluation_prompt(
    source_abstract: str, source_related_work: str, target_related_work: str, reverse_order: bool = False
) -> tuple[str, list[str]]:
    order = []
    if reverse_order:
        order = ["target", "source"]
        tmp = source_related_work
        source_related_work = target_related_work
        target_related_work = tmp
    else:
        order = ["source", "target"]

    return (
        f"""
    Source Abstract:
    {source_abstract}

    Related Works Section A:
    {source_related_work}

    Related Works Section B:
    {target_related_work}

    Objective:
    Evaluate which related works section better complements the source abstract provided.

    Consider factors such as comprehensiveness, clarity of writing, relevance, etc. when making your decision.
    If invalid citations occur, consider the information to be invalid (or even completely false!).

    Exclusively respond with your choice of rating from one of the following options:
        •	Section A
        •	Section B
        •	Tie

    Do not include anything else in your output.
    """,
        order,
    )


def generate_related_work_score_prompt(source_abstract: str, related_work: str) -> str:
    return f"""
    Source Abstract:
    {source_abstract}

    Related Works Section:
    {related_work}

    Objective:
    Evaluate this related works section with regard to the source abstract provided.

    Consider factors such as comprehensiveness, clarity of writing, relevance, etc. when making your decision.
    If invalid citations occur, consider the information to be invalid (or even completely false).

    Exclusively respond with your choice of rating. For this purpose you can assign a score from 0-10 where 0 is worst and 10 is best.

    - **0**: Completely irrelevant, unclear, or inaccurate.  
    *Example*: The section does not address the Source Abstract's topics and contains multiple invalid citations.  

    - **5**: Somewhat relevant but lacks comprehensiveness, clarity or relevance.  
    *Example*: The section references a few relevant works but also includes irrelevant ones and has minor errors.  

    - **10**: Exceptionally relevant, comprehensive, clear, and accurate.  
     *Example*: The section thoroughly addresses all key topics, includes all relevant works, and is clearly written with no factual errors.

    Do not include anything else in your output.
    """
