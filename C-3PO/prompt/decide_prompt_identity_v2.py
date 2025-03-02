NO_DOCUMENT_PROMPT = """{dataset_instructions}\nBased on your knowledge, answer the question:\n{question}"""

DOCUMENT_PROMPT_PREFIX = """You are a knowledgeable assistant. Please answer the following question by:
1. First reviewing the provided documents. Please extract the relevant information and ignore irrelevant information about the question.
2. Combining relevant information from the documents (if any) with your own knowledge to generate a response.
3. {dataset_instructions}"""

few_shot_template = """Here are some examples:
{examples}"""

DOCUMENT_PROMPT_SUFFIX = """Now, process the following question:
Existing documents: {documents}\n\nQuestion: {question}"""


DOCUMENT_PROMPT = DOCUMENT_PROMPT_PREFIX + "\n\n" + DOCUMENT_PROMPT_SUFFIX
DOCUMENT_PROMPT_FEW_SHOT = DOCUMENT_PROMPT_PREFIX + "\n\n" + few_shot_template + "\n\n" + DOCUMENT_PROMPT_SUFFIX

POPQA_DECISION = """Note that the question mainly asks about the object entity that holds a certain relationship with the given subject entity. There may be multiple correct answers. Make sure your response includes all correct answers and provides clear reasoning details followed by a concise conclusion."""
OTHERS_DECISION = """Note that the question may be compositional and require intermediate analysis to deduce the final answer. Make sure your response is grounded and provides clear reasoning details followed by a concise conclusion."""

instruct_for_each_dataset = {
    "PopQA": POPQA_DECISION,
    "2WikiMultiHopQA": OTHERS_DECISION,
    "NaturalQuestions": OTHERS_DECISION,
    "TriviaQA": OTHERS_DECISION,
    "hotpotqa": OTHERS_DECISION,
    "Musique": OTHERS_DECISION,
}

