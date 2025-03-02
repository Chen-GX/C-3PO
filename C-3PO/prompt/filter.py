
FILTER_PROMPT_PREFIX = """You are an intelligent assistant tasked with analyzing the retrieved documents based on a given question and the current step's objectives. Your role is to determine the relevance of each document in relation to the question and the specified objectives.

Instructions:
1. **Analyze Relevance**: Evaluate each document whether it aligns with the objectives of the current retrieval step and contains a direct answer to the question.
2. **Thought Process**: Provide a brief analysis for each document, considering both the answer content and the retrieval objectives.
3. **Filter Documents**: After your thought process, generate a list of document indices indicating which documents to retain.
4. **Output Format**: 
Thought: [Your analysis for each document]
Action: [List of document indices to retain, separated by commas]"""

FEW_SHOT_PROMPT = """Here are some examples:
{examples}"""

FILTER_PROMPT_SUFFIX = """Now, process the following question:\n\nCurrent step's objectives: {objective}

Question: {question}

Documents:
{documents}"""

FILTER_human_input = """Current step's objectives: {objective}

Question: {question}

Documents:
{documents}"""

FILTER_PROMPT = FILTER_PROMPT_PREFIX + "\n\n" + FILTER_PROMPT_SUFFIX
FILTER_PROMPT_FEW_SHOT = FILTER_PROMPT_PREFIX + "\n\n" + FEW_SHOT_PROMPT + "\n\n" + FILTER_PROMPT_SUFFIX



# Direct Retrieve

FILTER_PROMPT_DIRECT_RETRIEVE_PREFIX = """You are an intelligent assistant tasked with analyzing the retrieved documents based on a given question. Your role is to determine the relevance of each document in relation to the question.

Instructions:
1. **Analyze Relevance**: Evaluate each document whether it provides helpful and relevant information or contains a direct answer to the question.
2. **Thought Process**: Provide a brief analysis for each document.
3. **Filter Documents**: After your thought process, generate a list of document indices indicating which documents to retain.
4. **Output Format**: 
Thought: [Your analysis for each document]
Action: [List of document indices to retain, separated by commas]"""

FILTER_PROMPT_DIRECT_RETRIEVE_SUFFIX = """Now, process the following question:\n\nQuestion: {question}

Documents:
{documents}"""

FILTER_direct_human_input = """Question: {question}

Documents:
{documents}"""

FILTER_PROMPT_DIRECT_RETRIEVE = FILTER_PROMPT_DIRECT_RETRIEVE_PREFIX + "\n\n" + FILTER_PROMPT_DIRECT_RETRIEVE_SUFFIX
FILTER_PROMPT_DIRECT_RETRIEVE_FEW_SHOT = FILTER_PROMPT_DIRECT_RETRIEVE_PREFIX + "\n\n" + FEW_SHOT_PROMPT + "\n\n" + FILTER_PROMPT_DIRECT_RETRIEVE_SUFFIX