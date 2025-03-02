evaluation_prompt_prefix = """You are a precise answer validator. Your task is to compare the predicted answer with a set of acceptable correct answers and determine if the prediction matches any of them.

Input format:
Question: [The question text]
Correct Answers: [Array or list of acceptable correct answers]
Predicted Answer: [The answer to be evaluated]

Rules:
1. Consider semantic equivalence, not just exact string matching
2. Ignore minor differences in formatting, spacing, or capitalization
3. For numerical answers, consider acceptable margin of error if applicable
4. For text answers, focus on the core meaning rather than exact wording
5. The predicted answer is considered correct if it matches ANY ONE of the provided correct answers
6. The matching can be exact or semantically equivalent to any of the correct answers
7. Return only "True" if the predicted answer is correct, or "False" if it is incorrect."""

few_shot_template = """Here are some examples:
{examples}"""

evaluation_prompt_suffix = """Now, process the following question:
Question: {question}
Correct Answer: {true_answer}
Predicted Answer: {long_answer}\n"""


evaluation_prompt = evaluation_prompt_prefix + "\n\n" + evaluation_prompt_suffix
evaluation_prompt_few_shot = evaluation_prompt_prefix + "\n\n" + few_shot_template + "\n\n" + evaluation_prompt_suffix



if __name__ == "__main__":
    query = evaluation_prompt.format(
        question="When was Philip, Count Of Egmont's father born?",
        true_answer="['18 November 1522']",
        long_answer="The documents provided do not contain information about Philip, Count of Egmont. However, they do provide information about Lamoral, Count of Egmont, who was born on November 18, 1522. Lamoral's father, John IV, is mentioned, but his birthdate is not provided. Therefore, it is not possible to determine when Philip, Count of Egmont's father was born based on the given documents. It should be noted that there seems to be a confusion in the question, as the documents only provide information about Lamoral, Count of Egmont, and not Philip, Count of Egmont.",
    )
    print()

    "Philip, Count of Egmont's father was Lamoral, Count of Egmont. According to Document 1 and Document 3, Lamoral, Count of Egmont was born on November 18, 1522. Therefore, Philip, Count of Egmont's father was born on November 18, 1522."