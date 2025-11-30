from dotenv import load_dotenv
from modules.llm.llm_engine_interface import LLMInterface
import re
from modules.llm.llm_reasoning_interface import LLMReasoningInterface

load_dotenv(override=True)

class EnhancedLLMReasoning(LLMReasoningInterface):
    """
    Enhanced reasoning model that specializes in explaining nuanced fact-check results,
    particularly for contested claims or questions with multiple valid but conflicting data points.
    """
    
    def __init__(self, llm: LLMInterface):
        self.llm = llm.build()

    def call_llm(self, prompt, temperature=0.0):
        """Call LLM with adjustable temperature for different reasoning stages"""
        return self.llm.raw_messages(
            messages=[
                {"role": "system", "content": "You are a precise reasoning assistant specializing in fact verification analysis."},
                {"role": "user", "content": prompt}
            ],
        ).strip()

    def parse_fact_check_input(self, input_text, is_question=False):
        """
        Parse structured input from fact-checking results
        
        Args:
            input_text (str): The input text to parse
            is_question (bool): Flag indicating whether the input is a question
        """
        patterns = {
            'claim': r'Claim:\s*(.*?)(?:\n|$)',
            'verdict': r'Verdict:\s*(.*?)(?:\n|$)',
            'score': r'Score:\s*(.*?)(?:/100)?(?:\n|$)',
            'citations': r'Citations:\s*([\s\S]*?)(?:\n\n|$)',
            'question': r'Question:\s*([\s\S]*?)(?:\n\n|$)'
        }
        is_question = patterns['question'] in input_text or is_question
        
        result = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, input_text)
            if match:
                result[key] = match.group(1).strip()
                
        # Parse citations into a list
        if 'citations' in result:
            result['citation_list'] = [c.strip() for c in result['citations'].split('\n') if c.strip()]
        
        # Add the is_question flag to the result
        result['is_question'] = is_question
            
        return result

    def identify_input_type(self, text, is_question=False):
        """
        Identify the type of input (claim or question) for specialized reasoning
        
        Args:
            text (str): The text to analyze
            is_question (bool): Flag indicating whether the input is a question
        """
        input_label = "Question" if is_question else "Claim"
        
        prompt = f"""Identify the type of {input_label.lower()} below:

{input_label}: "{text}"

Choose ONE category that best describes this {input_label.lower()}:
1. Statistical/Numerical (involves specific numbers, statistics, measurements)
2. Temporal/Historical (involves dates, timelines, historical events)
3. Causal (involves cause and effect relationships)
4. Categorical (involves classification or categorization)
5. Comparative (involves comparison between entities)
6. Attributive (involves attributing properties/qualities to something)
7. Existential (involves existence of something)

Type:"""

        response = self.call_llm(prompt)
        # Extract just the type label
        input_type = response.strip()
        if "Statistical" in input_type or "Numerical" in input_type:
            return "statistical"
        elif "Temporal" in input_type or "Historical" in input_type:
            return "temporal"
        elif "Causal" in input_type:
            return "causal"
        elif "Comparative" in input_type:
            return "comparative"
        else:
            return "general"

    def analyze_contested_input(self, parsed_input):
        """Specialized analysis for contested claims/questions with conflicting evidence"""
        text = parsed_input.get('claim', '')
        citations = parsed_input.get('citation_list', [])
        is_question = parsed_input.get('is_question', False)
        
        input_label = "Question" if is_question else "Claim"
        input_type = self.identify_input_type(text, is_question)
        
        if input_type == "statistical":
            # For statistical claims/questions, extract and compare the numbers
            prompt = f"""Analyze the following statistical {input_label.lower()} that has been marked as contested:

{input_label}: "{text}"

Evidence:
{chr(10).join([f"- {c}" for c in citations])}

Perform the following analysis:
1. Extract all numerical values from the {input_label.lower()} and evidence
2. Identify specifically which metrics are being compared
3. Explain why these statistics appear to contradict each other
4. Consider possible explanations: different time periods, different metrics, different data sources
5. Determine what makes this {input_label.lower()} technically "contested" rather than supported or refuted
{f"6. Identify the specific answer to the question that is contested" if is_question else ""}

Detailed analysis:"""

        elif input_type == "temporal":
            # For temporal claims/questions, focus on timelines and chronology
            prompt = f"""Analyze the following temporal/historical {input_label.lower()} that has been marked as contested:

{input_label}: "{text}"

Evidence:
{chr(10).join([f"- {c}" for c in citations])}

Perform the following analysis:
1. Extract all dates, time periods, or chronological information from the {input_label.lower()} and evidence
2. Identify specifically where there are temporal inconsistencies
3. Consider how different time frames might make {f"the answer to this question" if is_question else "this claim"} simultaneously true and false
4. Determine what makes this {input_label.lower()} technically "contested" rather than supported or refuted
{f"5. Identify the specific answer to the question that is contested" if is_question else ""}

Detailed analysis:"""

        elif input_type == "comparative":
            # For comparative claims/questions
            prompt = f"""Analyze the following comparative {input_label.lower()} that has been marked as contested:

{input_label}: "{text}"

Evidence:
{chr(10).join([f"- {c}" for c in citations])}

Perform the following analysis:
1. Identify what entities or metrics are being compared in the {input_label.lower()}
2. Analyze how the evidence presents different comparative relationships
3. Explain how the comparison might be valid from multiple perspectives
4. Determine what makes this {input_label.lower()} technically "contested" rather than supported or refuted
{f"5. Identify the specific answer to the question that is contested" if is_question else ""}

Detailed analysis:"""

        else:
            # General approach for other claim/question types
            prompt = f"""Analyze the following {input_label.lower()} that has been marked as contested:

{input_label}: "{text}"

Evidence:
{chr(10).join([f"- {c}" for c in citations])}

Perform the following analysis:
1. Identify the core {f"inquiry" if is_question else "assertion"} in the {input_label.lower()}
2. Explain how the evidence {f"leads to contradictory answers" if is_question else "both supports and contradicts aspects of the claim"}
3. Highlight the specific points of contention or contradiction
4. Consider if the contested nature comes from ambiguity, partial truth, or context
5. Determine what makes this {input_label.lower()} technically "contested" rather than supported or refuted
{f"6. Identify the specific answer to the question that is contested" if is_question else ""}

Detailed analysis:"""

        return self.call_llm(prompt, temperature=0.1)  # Slight creativity for analysis

    def reconcile_evidence(self, text, analysis, citations, is_question=False):
        """Attempt to reconcile apparently conflicting evidence"""
        input_label = "Question" if is_question else "Claim"
        
        prompt = f"""Based on the following analysis of a contested {input_label.lower()}, reconcile the conflicting evidence:

{input_label}: "{text}"

Analysis:
{analysis}

Evidence:
{chr(10).join([f"- {c}" for c in citations])}

Provide a reconciliation that explains:
1. How these apparently conflicting pieces of evidence can be understood together
2. What additional context would help resolve this contestation
3. If there are temporal, definitional, or methodological differences that explain the conflict
4. The most precise and accurate way to understand this information
{f"5. The most accurate answer to the question given the conflicting evidence" if is_question else ""}

Reconciliation:"""

        return self.call_llm(prompt)

    def generate_verdict_explanation(self, parsed_input):
        """Generate a comprehensive explanation for the verdict"""
        text = parsed_input.get('claim', '')
        verdict = parsed_input.get('verdict', '')
        score = parsed_input.get('score', '')
        citations = parsed_input.get('citation_list', [])
        is_question = parsed_input.get('is_question', False)
        
        input_label = "Question" if is_question else "Claim"
        
        # For contested claims/questions, use specialized reasoning
        if verdict.lower() == "contested":
            analysis = self.analyze_contested_input(parsed_input)
            reconciliation = self.reconcile_evidence(text, analysis, citations, is_question)
            
            final_prompt = f"""Provide a final explanation for this fact-check result:

{input_label}: "{text}"
Verdict: {verdict}
Score: {score}/100

Your explanation should:
1. Clearly explain why {f"the answer to this question" if is_question else "this claim"} is contested, not supported or refuted
2. Reference the specific evidence that creates this contestation
3. Explain what would be needed to resolve this contestation
4. Be precise about which aspects are true and which are misleading
5. Use language that is clear, balanced, and educational
{f"6. Provide the most accurate answer possible given the conflicting evidence" if is_question else ""}

Based on the analysis:
{analysis}

And the reconciliation:
{reconciliation}

Final explanation:"""
            
            return self.call_llm(final_prompt)
            
        else:
            # For non-contested verdicts, use a simpler approach
            prompt = f"""Explain the following fact-check result:

{input_label}: "{text}"
Verdict: {verdict}
Score: {score}/100
Evidence:
{chr(10).join([f"- {c}" for c in citations])}

Provide a clear, concise explanation for why this verdict was reached, referencing specific evidence.
{f"Include a direct answer to the question based on the evidence." if is_question else ""}

Explanation:"""
            
            return self.call_llm(prompt)

    def reasoning_agent(self, input_text, is_question=False):
        """
        Main entry point for reasoning about fact-check results
        
        Args:
            input_text (str): The input text to process
            is_question (bool): Flag indicating whether the input is a question
        """
        print("**Enhanced LLM Reasoning Agent Invoked**")
        parsed_input = self.parse_fact_check_input(input_text, is_question)
        
        if not parsed_input.get('verdict'):
            # If no verdict found, use general reasoning
            return self.call_llm(input_text)
        
        verdict = parsed_input.get('verdict', '').lower()
        if verdict == "not enough evidence":
            if is_question:
                return "We don't have enough evidence and data to answer this question."
            else:
                return "We don't have enough evidence and data for this claim."
            
        # For all other verdicts, generate a specialized explanation
        return self.generate_verdict_explanation(parsed_input)


class NBA_Statistics_Reasoner(EnhancedLLMReasoning):
    """
    A specialized reasoner for NBA statistics that understands the nuances of sports statistics
    and can explain apparent contradictions in player data across different seasons, teams, or metrics.
    """
    
    def __init__(self, llm: LLMInterface):
        super().__init__(llm)
        
    def identify_statistical_pattern(self, text, citations, is_question=False):
        """Identify statistical patterns in NBA data"""
        input_label = "Question" if is_question else "Claim"
        
        prompt = f"""Analyze the following NBA statistical {input_label.lower()}:

{input_label}: "{text}"

Evidence:
{chr(10).join([f"- {c}" for c in citations])}

Identify:
1. The player(s) mentioned
2. The specific statistics referenced (points, rebounds, assists, etc.)
3. Any seasons or teams mentioned
4. Any qualifiers or conditions (regular season, playoffs, career average, etc.)
{f"5. The specific statistical question being asked" if is_question else ""}

Statistical pattern analysis:"""

        return self.call_llm(prompt)
        
    def analyze_contested_input(self, parsed_input):
        """Override to provide NBA-specific analysis for contested claims/questions"""
        text = parsed_input.get('claim', '')
        citations = parsed_input.get('citation_list', [])
        is_question = parsed_input.get('is_question', False)
        
        input_label = "Question" if is_question else "Claim"
        
        # Check if text involves NBA statistics
        if "NBA" in text or any("NBA" in c for c in citations):
            statistical_analysis = self.identify_statistical_pattern(text, citations, is_question)
            
            prompt = f"""Analyze the following contested NBA statistical {input_label.lower()}:

{input_label}: "{text}"

Evidence:
{chr(10).join([f"- {c}" for c in citations])}

Statistical Analysis:
{statistical_analysis}

Explain in detail:
1. The specific statistical discrepancies between {f"what is being asked and the evidence" if is_question else "the claim and evidence"}
2. If the statistics reflect different seasons, teams, or contexts
3. How career averages vs. season averages might affect interpretation
4. How different statistical qualifying criteria might apply (min. games played, etc.)
5. Whether the statistics are regular season, playoff, or combined
6. How these factors make {f"the answer to this question" if is_question else "this claim"} technically "contested" rather than clearly true or false
{f"7. The most accurate answer to the question given the conflicting evidence" if is_question else ""}

Detailed NBA statistical analysis:"""

            return self.call_llm(prompt, temperature=0.1)
        
        # Fall back to general analysis for non-NBA claims/questions
        return super().analyze_contested_input(parsed_input)

    def reconcile_evidence(self, text, analysis, citations, is_question=False):
        """Override to provide NBA-specific reconciliation"""
        input_label = "Question" if is_question else "Claim"
        
        if "NBA" in text or any("NBA" in c for c in citations):
            prompt = f"""Based on the following analysis of a contested NBA statistical {input_label.lower()}, 
reconcile the apparently conflicting evidence:

{input_label}: "{text}"

NBA Statistical Analysis:
{analysis}

Evidence:
{chr(10).join([f"- {c}" for c in citations])}

Provide a reconciliation that explains:
1. How these statistics can be accurately understood in their proper contexts
2. Whether the differences are due to:
   - Different seasons being referenced
   - Career vs. individual season averages
   - Regular season vs. playoff statistics
   - Different teams or roles
   - Changes in playing time or usage
3. {f"The most accurate and precise answer that could be given to this question" if is_question else "The most accurate and precise statement that could be made about this player's statistics"}
4. How a sports analyst would properly contextualize these numbers

NBA statistical reconciliation:"""

            return self.call_llm(prompt)
            
        # Fall back to general reconciliation for non-NBA claims/questions
        return super().reconcile_evidence(text, analysis, citations, is_question)


class Temporal_Context_Reasoner(EnhancedLLMReasoning):
    """
    A specialized reasoner that understands how time and context affect fact verification,
    particularly useful for claims/questions that may have been true in one period but not another.
    """
    
    def __init__(self, llm: LLMInterface):
        super().__init__(llm)
        
    def extract_temporal_context(self, text, citations, is_question=False):
        """Extract temporal context from claims/questions and evidence"""
        input_label = "Question" if is_question else "Claim"
        
        prompt = f"""Extract all temporal information from this {input_label.lower()} and evidence:

{input_label}: "{text}"

Evidence:
{chr(10).join([f"- {c}" for c in citations])}

Identify:
1. All specific dates, years, seasons, or time periods mentioned
2. Any implicit temporal references
3. The chronological order of relevant events
4. Any changes over time that would affect {f"the answer to this question" if is_question else "the claim's validity"}
{f"5. How time affects the possible answers to this question" if is_question else ""}

Temporal context analysis:"""

        return self.call_llm(prompt)
        
    def analyze_contested_input(self, parsed_input):
        """Override to provide temporal-specific analysis"""
        text = parsed_input.get('claim', '')
        citations = parsed_input.get('citation_list', [])
        is_question = parsed_input.get('is_question', False)
        
        input_label = "Question" if is_question else "Claim"
        
        # Extract temporal information first
        temporal_context = self.extract_temporal_context(text, citations, is_question)
        
        prompt = f"""Analyze this temporally contested {input_label.lower()}:

{input_label}: "{text}"

Evidence:
{chr(10).join([f"- {c}" for c in citations])}

Temporal Context:
{temporal_context}

Explain in detail:
1. How different time periods affect {f"the answer to this question" if is_question else "the truth of this claim"}
2. Whether {f"the answer" if is_question else "the claim"} was true during some periods and false during others
3. If there are relevant changes over time that affect {f"how this question should be answered" if is_question else "the claim's validity"}
4. How these temporal factors make {f"the answer" if is_question else "the claim"} "contested" rather than simply true or false
{f"5. The most accurate time-sensitive answer to this question" if is_question else ""}

Temporal analysis:"""

        return self.call_llm(prompt, temperature=0.1)

class Multi_Perspective_Reasoner(EnhancedLLMReasoning):
    """
    A reasoner that considers multiple valid perspectives to explain
    why claims/questions might be contested rather than simply true or false.
    """
    
    def __init__(self, llm: LLMInterface):
        super().__init__(llm)
        
    def identify_perspectives(self, text, citations, is_question=False):
        """Identify different valid perspectives on the claim or question"""
        input_label = "Question" if is_question else "Claim"
        
        prompt = f"""Identify different valid perspectives on this {input_label.lower()}:

{input_label}: "{text}"

Evidence:
{chr(10).join([f"- {c}" for c in citations])}

Identify at least 2-3 different perspectives from which this {input_label.lower()} could be evaluated,
considering differences in:
1. Definitions of key terms
2. Methodological approaches
3. Contextual assumptions
4. Frames of reference
5. Evaluation criteria
{f"6. How these perspectives would lead to different answers" if is_question else ""}

Multiple perspectives:"""

        return self.call_llm(prompt, temperature=0.2)  # More creative for perspective generation
        
    def analyze_contested_input(self, parsed_input):
        """Override to provide multi-perspective analysis"""
        text = parsed_input.get('claim', '')
        citations = parsed_input.get('citation_list', [])
        is_question = parsed_input.get('is_question', False)
        
        input_label = "Question" if is_question else "Claim"
        
        # Identify different perspectives
        perspectives = self.identify_perspectives(text, citations, is_question)
        
        prompt = f"""Analyze this contested {input_label.lower()} from multiple perspectives:

{input_label}: "{text}"

Evidence:
{chr(10).join([f"- {c}" for c in citations])}

Perspectives:
{perspectives}

Explain in detail:
1. How different perspectives lead to different {f"answers to this question" if is_question else "evaluations of this claim"}
2. Why reasonable people might disagree about {f"how to answer this question" if is_question else "this claim's validity"}
3. What underlying assumptions or definitions create this disagreement
4. How these multiple valid perspectives make {f"the answer to this question" if is_question else "this claim"} "contested" rather than simply true or false
{f"5. What the most balanced answer would be that acknowledges these different perspectives" if is_question else ""}

Multi-perspective analysis:"""

        return self.call_llm(prompt, temperature=0.1)
