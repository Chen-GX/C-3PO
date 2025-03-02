
DECIDE_PROMPT_QP = """You are tasked with selecting the most appropriate prompt from a list of candidates based on a given question and passages. Your role is to evaluate the suitability of each prompt in relation to the provided context.

Instructions:
1. You will receive a question, a set of passages, and a list of candidate prompts.
2. Assess each prompt for its relevance and suitability in addressing the given question, taking into account the provided passages if they are available.
3. **Do not** attempt to answer the question or modify the prompts; your responsibility is solely to evaluate and select.
4. Output the index of the prompt that you believe is the best fit for the provided question and passages. The output should be in the form of a single integer.

Context:
- Question: {questions}
- Passages:
{passages}
- Candidate Prompts:
{prompts}

Considerations:
- Evaluate the prompts based on clarity, relevance, and potential effectiveness in guiding an LLM to generate a high-quality response to the question.
- Consider how well each prompt aligns with the main themes or concepts in the provided question and passages.

Please provide your output below:"""


DECIDE_PROMPT_Q = """Your task is to select the most appropriate prompt based on a given question. You need to evaluate the suitability of each prompt to effectively address the provided question.

Instructions:
1. You will receive a question and a list of candidate prompts.
2. Assess each prompt for its relevance and effectiveness in addressing the given question.
3. **Do not** attempt to answer the question or modify the prompts; your responsibility is solely to evaluate and select.
4. Output the index of the prompt that you believe is the best fit for the provided question. The output should be in the form of a single integer enclosed in square brackets.

Context:
- Question: {questions}
- Candidate Prompts:
{prompts}

Considerations:
- Evaluate the prompts on the basis of clarity, relevance, and their potential effectiveness in guiding an LLM to generate a high-quality response to the question.

Please provide your output below:"""
