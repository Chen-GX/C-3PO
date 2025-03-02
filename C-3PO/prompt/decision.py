prompt_decision_making_prefix = """You are an intelligent assistant tasked with evaluating whether a given question requires further information through retrieval or needs planning to arrive at an accurate answer. You will have access to a large language model (LLM) for planning or answering the question and a retrieval system to provide relevant information about the query.

Instructions:
1. **Evaluate the Question**: Assess whether a precise answer can be provided based on the existing knowledge of LLM. Consider the specificity, complexity, and clarity of the question.
2. **Decision Categories:**
    - If the question is complex and requires a planning phase before retrieval, your response should be:
    [Planning]
    - If the question requests specific information that you believe the LLM does not possess or pertains to recent events or niche topics outside LLM's knowledge scope, format your response as follows: 
    [Retrieval] `YOUR QUERY HERE`
    - If you think the LLM can answer the question without additional information, respond with:
    [No Retrieval]
3. **Focus on Assessment**: Avoid providing direct answers to the questions. Concentrate solely on determining the necessity for retrieval or planning.{dataset_instructions}"""  # {dataset_instructions}

prompt_decision_making_suffix_few_shot = """Now, process the following question:\n\nQuestion: {question}\n"""

decision_human_input = "Question: {question}\n"

few_shot_template = """Here are some examples:
{examples}"""

prompt_decision_making = prompt_decision_making_prefix + "\n\n" + prompt_decision_making_suffix_few_shot

prompt_decision_making_few_shot = prompt_decision_making_prefix + "\n\n" + few_shot_template + "\n\n" + prompt_decision_making_suffix_few_shot

POPQA_DECISION = """\n4. Keep in mind that the question mainly asks about the object entity that holds a certain relationship with the given subject entity. There may be multiple correct answers."""
OTHERS_DECISION = """\n4. Keep in mind that the question may be compositional."""

instruct_for_each_dataset = {
    "PopQA": POPQA_DECISION,
    "2WikiMultiHopQA": OTHERS_DECISION,
    "NaturalQuestions": OTHERS_DECISION,
    "TriviaQA": OTHERS_DECISION,
    "hotpotqa": OTHERS_DECISION,
    "Musique": OTHERS_DECISION,
}




if __name__ == "__main__":
    print(prompt_decision_making_few_shot)
