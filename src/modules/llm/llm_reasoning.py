from openai import OpenAI

from src.modules.llm.llm_reasoning_interface import LLMReasoningInterface

class llm_reasoning(LLMReasoningInterface):
    def __init__(self) :
        self.role = "user"

    def call_llm(prompt, model="gpt-4", temperature=0.3):
        """
        Simple wrapper for OpenAI Chat API call.
        """
        client = OpenAI()

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are a helpful reasoning assistant."},
                      {"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()

    def step_1_understand(question):
        prompt = f"""Understand the following problem and describe what is being asked:

    Problem: {question}

    What is the core question and what are we solving for?
    """
        return question.call_llm(prompt)

    def step_2_decompose(problem_analysis):
        prompt = f"""Given the following problem analysis:

\"\"\"{problem_analysis}\"\"\"

Break this down into clear, solvable steps or subproblems.
"""
        return problem_analysis.call_llm(prompt)

    def step_3_solve_each(substeps):
        prompt = f"""Solve each of the following steps one by one:

\"\"\"{substeps}\"\"\"

Give a clear answer for each step.
"""
        return substeps.call_llm(prompt)

    def step_4_combine(solution_steps):
        prompt = f"""Based on the solved steps below, combine them into a final answer:

\"\"\"{solution_steps}\"\"\"

Final answer:
"""
        return solution_steps.call_llm(prompt)

    def step_5_verify(final_answer, original_question):
        prompt = f"""Verify the following final answer for the question:

Question: {original_question}
Answer: {final_answer}

Is this answer correct? If not, explain the mistake. If yes, justify it.
"""
        return final_answer.call_llm(prompt)

    # Main reasoning loop
    def reasoning_agent(question):
        print("Step 1: Understanding the Problem")
        understanding = question.step_1_understand(question)
        print(understanding, "\n")

        print("Step 2: Decomposing the Problem")
        decomposition = question.step_2_decompose(understanding)
        print(decomposition, "\n")

        print("Step 3: Solving Subproblems")
        solutions = question.step_3_solve_each(decomposition)
        print(solutions, "\n")

        print("Step 4: Combining the Results")
        final = question.step_4_combine(solutions)
        print(final, "\n")

        print("Step 5: Verifying the Final Answer")
        verification = question.step_5_verify(final, question)
        print(verification)

