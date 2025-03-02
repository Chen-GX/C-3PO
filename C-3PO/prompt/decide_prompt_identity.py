NO_DOCUMENT_PROMPT = """{dataset_instructions}\nBased on your knowledge, answer the question:\n{question}"""

DOCUMENT_PROMPT = """Existing documents: {documents}\n\n{dataset_instructions}\nBased on your knowledge and the provided information, answer the question:\n{question}"""

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


