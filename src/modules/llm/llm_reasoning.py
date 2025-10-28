from dotenv import load_dotenv
from modules.llm.llm_engine_interface import LLMInterface
load_dotenv(override=True)

from modules.llm.llm_reasoning_interface import LLMReasoningInterface

class llm_reasoning(LLMReasoningInterface):
    def __init__(self, llm: LLMInterface):
        self.llm = llm.build()

    def call_llm(self, prompt):
        return self.llm.raw_messages(
            messages=[
                {"role": "system", "content": "You are a helpful reasoning assistant."},
                {"role": "user", "content": prompt}
            ],
        ).strip()

    def step_1_understand(self, question):
        prompt = f"""Understand the following problem and describe what is being asked:

Problem: {question}

What is the core question and what are we solving for?
"""
        return self.call_llm(prompt)

    def step_2_decompose(self, problem_analysis):
        prompt = f"""Given the following problem analysis:

\"\"\"{problem_analysis}\"\"\"

Break this down into clear, solvable steps or subproblems.
"""
        return self.call_llm(prompt)

    def step_3_solve_each(self, substeps):
        prompt = f"""Solve each of the following steps one by one:

\"\"\"{substeps}\"\"\"

Give a clear answer for each step.
"""
        return self.call_llm(prompt)

    def step_4_combine(self, solution_steps):
        prompt = f"""Based on the solved steps below, combine them into a final answer:

\"\"\"{solution_steps}\"\"\"

Final answer:
"""
        return self.call_llm(prompt)

    def step_5_verify(self, final_answer, original_question):
        prompt = f"""Verify the following final answer for the question:

Question: {original_question}
Answer: {final_answer}

Is this answer correct? If not, explain the mistake. If yes, justify it.
"""
        return self.call_llm(prompt)

    def reasoning_agent(self, question):
        understanding = self.step_1_understand(question)
        decomposition = self.step_2_decompose(understanding)
        solutions = self.step_3_solve_each(decomposition)
        final = self.step_4_combine(solutions)
        verification = self.step_5_verify(final, question)
        
        return verification