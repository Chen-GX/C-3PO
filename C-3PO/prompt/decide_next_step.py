decide_next_step_prefix = """You are an intelligent assistant tasked with determining the next appropriate action based on the provided existing documents, plan, and question. You have access to a large language model (LLM) for answering question and a retrieval system for gathering additional documents. Your objective is to decide whether to write a query for retrieving relevant documents or to generate a comprehensive answer using the LLM based on the existing documents and plan.

Instructions:
1. **Evaluate Existing Documents**: Assess the existing documents to determine if it is sufficient to answer the question.
2. **Follow the Plan**: Understand the next steps outlined in the plan.
3. **Decision Categories:**
    - If the existing documents is insufficient and requires additional retrieval, respond with:
        [Retrieval] `YOUR QUERY HERE`
    - If the existing documents is adequate to answer the question, respond with:
        [LLM]
4. **Focus on Action**: Do not answer the question directly; concentrate on identifying the next appropriate action based on the existing documents, plan, and question.
5. **Output Format**: 
Thought: [Your analysis for current situation (need retrieval for additional informations or use LLM to answer)]
Action: [Your decision based on the analysis (Retrieval or LLM)]"""

decide_next_step_suffix = """Now, process the following question:\n\nExisting Documents: {existing_documents}

Plan: {planning}

Question: {question}\n"""

decide_next_step_human_input = """Existing Documents: {existing_documents}

Plan: {planning}

Question: {question}\n"""

few_shot_template = """Here are some examples:
{examples}"""

decide_next_step = decide_next_step_prefix + "\n\n" + decide_next_step_suffix
decide_next_step_few_shot = decide_next_step_prefix + "\n\n" + few_shot_template + "\n\n" + decide_next_step_suffix


# 单纯LLM
decide_next_step_LLM_prefix = """You are an intelligent assistant assigned to analyze useful information from the existing documents and plan for responding to a question.

Instructions:
1. **Evaluate Existing Documents**: Thoroughly review the provided documents to extract useful information relevant to the question.
2. **Decision Categories:**
    - After your analysis, you should respond with:
        [LLM]
3. **Focus on Action**: Focus solely on identifying and analyzing relevant information rather than answering the question directly.
4. **Output Format**: 
Thought: [Provide a detailed analysis outlining the useful information.]
Action: [LLM]"""


decide_next_step_LLM = decide_next_step_LLM_prefix + "\n\n" + decide_next_step_suffix
decide_next_step_LLM_few_shot = decide_next_step_LLM_prefix + "\n\n" + few_shot_template + "\n\n" + decide_next_step_suffix
