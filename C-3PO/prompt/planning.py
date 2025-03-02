prompt_planning_prefix = """You are an expert assistant tasked with analyzing the following question and formulating a detailed plan. You will utilize a retrieval system to gather relevant information in your planning. Your goal is to analysis the question and provide a structured sequence of actions to address it effectively.

Instructions:
1. **Question Analysis**: Identifying the core components of the question. Determine what key information we currently know and what additional information is needed through retrieval.
2. **Step By Step Planning**: Develop a detailed plan step by step. Focus on the planning process rather than providing direct answers.
3. **Focus on Planning**: Keep your response clear and structured, concentrating solely on the analysis and planning aspects.{dataset_instructions}"""


prompt_planning_suffix_few_shot = """Now, process the following question:
Question: {question}\n"""

few_shot_template = """Here are some examples:
{examples}"""

prompt_llm_planning = prompt_planning_prefix + "\n\n" + prompt_planning_suffix_few_shot

prompt_llm_planning_few_shot = prompt_planning_prefix + "\n\n" + few_shot_template + "\n\n" + prompt_planning_suffix_few_shot


ASQA_DECISION = """\n4. Keep in mind that the question may be ambiguous and may have multiple correct answers. Ensure that your planning outlines are clear, especially for ambiguous questions."""
POPQA_DECISION = """\n4. Keep in mind that the question mainly asks about the object entity that holds a certain relationship with the given subject entity. There may be multiple correct answers. Ensure that your planning outlines are clear, especially for ambiguous questions."""
OTHERS_DECISION = """\n4. Keep in mind that the question may be compositional and require intermediate analysis to deduce the final answer. Ensure that your planning outlines are clear."""

instruct_for_each_dataset = {
    "ASQA": ASQA_DECISION,
    "PopQA": POPQA_DECISION,
    "2WikiMultiHopQA": OTHERS_DECISION,
    "NaturalQuestions": OTHERS_DECISION,
    "TriviaQA": OTHERS_DECISION,
}