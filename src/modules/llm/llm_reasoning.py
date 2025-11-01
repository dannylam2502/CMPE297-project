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

    def extract_components(self, verification, final, solutions):
        """
        Extract verdict, score, features, and citations from the verification result.
        Returns a dictionary with these components.
        """
        # Extract verdict (True/False/Inconclusive)
        if "correct" in verification.lower() and "yes" in verification.lower():
            verdict = True
        elif "not correct" in verification.lower() or "incorrect" in verification.lower():
            verdict = False
        else:
            verdict = "Inconclusive"
        
        # Calculate a confidence score (0-1)
        if verdict is True:
            confidence_indicators = ["definitely", "certainly", "absolutely", "clearly", "strongly"]
            score = 0.8  # Default high score for correct answers
            for indicator in confidence_indicators:
                if indicator in verification.lower():
                    score = min(1.0, score + 0.05)  # Boost score but cap at 1.0
        elif verdict is False:
            score = 0.2  # Default low score for incorrect answers
        else:
            score = 0.5  # Neutral score for inconclusive results
        
        # Extract features (key points from the reasoning)
        features = []
        # Extract from solutions (more detailed reasoning)
        solution_points = solutions.split("\n")
        for point in solution_points:
            if point.strip() and len(point.strip()) > 20:  # Non-empty and reasonably detailed
                features.append(point.strip())
        
        # Limit to most relevant features (max 5)
        features = features[:5]
        
        # Extract potential citations
        citations = []
        potential_citations = []
        
        # Look for patterns like "According to X" or "X states that"
        for section in [verification, final, solutions]:
            lines = section.split("\n")
            for line in lines:
                if "according to" in line.lower() or "states that" in line.lower() or "cited" in line.lower() or "reference" in line.lower():
                    potential_citations.append(line)
        
        # Process potential citations to extract the actual citation
        for citation in potential_citations:
            if len(citation) > 10:  # Reasonably sized citation
                citations.append(citation)
        
        # Limit to top citations (max 3)
        citations = citations[:3]
        
        return {
            "verdict": verdict,
            "score": round(score, 2),
            "features": features,
            "citations": citations
        }

    def reasoning_agent(self, question):
        understanding = self.step_1_understand(question)
        decomposition = self.step_2_decompose(understanding)
        solutions = self.step_3_solve_each(decomposition)
        final = self.step_4_combine(solutions)
        verification = self.step_5_verify(final, question)
        # Extract components and create the result dictionary
        result = self.extract_components(verification, final, solutions)
        
        # Include the full reasoning for reference
        result["understanding"] = understanding
        result["decomposition"] = decomposition
        result["solutions"] = solutions
        result["final"] = final
        result["verification"] = verification
        
        return result

