from dotenv import load_dotenv
from modules.llm.llm_engine_interface import LLMInterface
import re
from modules.llm.llm_reasoning_interface import LLMReasoningInterface
from typing import List, Dict, Any, Optional

load_dotenv(override=True)

class EnhancedLLMReasoning(LLMReasoningInterface):
    """
    Enhanced reasoning model that specializes in explaining nuanced fact-check results,
    particularly for contested claims with multiple valid but conflicting data points.
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

    def parse_fact_check_input(self, input_text):
        """Parse structured input from fact-checking results"""
        patterns = {
            'claim': r'Claim:\s*(.*?)(?:\n|$)',
            'verdict': r'Verdict:\s*(.*?)(?:\n|$)',
            'score': r'Score:\s*(.*?)(?:/100)?(?:\n|$)',
            'citations': r'Citations:\s*([\s\S]*?)(?:\n\n|$)'
        }
        
        result = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, input_text)
            if match:
                result[key] = match.group(1).strip()
                
        # Parse citations into a list
        if 'citations' in result:
            result['citation_list'] = [c.strip() for c in result['citations'].split('\n') if c.strip()]
            
        return result

    def identify_claim_type(self, claim):
        """Identify the type of claim for specialized reasoning"""
        prompt = f"""Identify the type of factual claim below:

Claim: "{claim}"

Choose ONE category that best describes this claim:
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
        claim_type = response.strip()
        if "Statistical" in claim_type or "Numerical" in claim_type:
            return "statistical"
        elif "Temporal" in claim_type or "Historical" in claim_type:
            return "temporal"
        elif "Causal" in claim_type:
            return "causal"
        elif "Comparative" in claim_type:
            return "comparative"
        else:
            return "general"

    def analyze_contested_claim(self, parsed_input):
        """Specialized analysis for contested claims with conflicting evidence"""
        claim = parsed_input.get('claim', '')
        citations = parsed_input.get('citation_list', [])
        
        claim_type = self.identify_claim_type(claim)
        
        if claim_type == "statistical":
            # For statistical claims, extract and compare the numbers
            prompt = f"""Analyze the following statistical claim that has been marked as contested:

Claim: "{claim}"

Evidence:
{chr(10).join([f"- {c}" for c in citations])}

Perform the following analysis:
1. Extract all numerical values from the claim and evidence
2. Identify specifically which metrics are being compared
3. Explain why these statistics appear to contradict each other
4. Consider possible explanations: different time periods, different metrics, different data sources
5. Determine what makes this claim technically "contested" rather than supported or refuted

Detailed analysis:"""

        elif claim_type == "temporal":
            # For temporal claims, focus on timelines and chronology
            prompt = f"""Analyze the following temporal/historical claim that has been marked as contested:

Claim: "{claim}"

Evidence:
{chr(10).join([f"- {c}" for c in citations])}

Perform the following analysis:
1. Extract all dates, time periods, or chronological information from the claim and evidence
2. Identify specifically where there are temporal inconsistencies
3. Consider how different time frames might make the claim simultaneously true and false
4. Determine what makes this claim technically "contested" rather than supported or refuted

Detailed analysis:"""

        elif claim_type == "comparative":
            # For comparative claims
            prompt = f"""Analyze the following comparative claim that has been marked as contested:

Claim: "{claim}"

Evidence:
{chr(10).join([f"- {c}" for c in citations])}

Perform the following analysis:
1. Identify what entities or metrics are being compared in the claim
2. Analyze how the evidence presents different comparative relationships
3. Explain how the comparison might be valid from multiple perspectives
4. Determine what makes this claim technically "contested" rather than supported or refuted

Detailed analysis:"""

        else:
            # General approach for other claim types
            prompt = f"""Analyze the following claim that has been marked as contested:

Claim: "{claim}"

Evidence:
{chr(10).join([f"- {c}" for c in citations])}

Perform the following analysis:
1. Identify the core assertion in the claim
2. Explain how the evidence both supports and contradicts aspects of the claim
3. Highlight the specific points of contention or contradiction
4. Consider if the contested nature comes from ambiguity, partial truth, or context
5. Determine what makes this claim technically "contested" rather than supported or refuted

Detailed analysis:"""

        return self.call_llm(prompt, temperature=0.1)  # Slight creativity for analysis

    def reconcile_evidence(self, claim, analysis, citations):
        """Attempt to reconcile apparently conflicting evidence"""
        prompt = f"""Based on the following analysis of a contested claim, reconcile the conflicting evidence:

Claim: "{claim}"

Analysis:
{analysis}

Evidence:
{chr(10).join([f"- {c}" for c in citations])}

Provide a reconciliation that explains:
1. How these apparently conflicting pieces of evidence can be understood together
2. What additional context would help resolve this contestation
3. If there are temporal, definitional, or methodological differences that explain the conflict
4. The most precise and accurate way to understand this information

Reconciliation:"""

        return self.call_llm(prompt)

    def generate_verdict_explanation(self, parsed_input):
        """Generate a comprehensive explanation for the verdict"""
        claim = parsed_input.get('claim', '')
        verdict = parsed_input.get('verdict', '')
        score = parsed_input.get('score', '')
        citations = parsed_input.get('citation_list', [])
        
        # For contested claims, use specialized reasoning
        if verdict.lower() == "contested":
            analysis = self.analyze_contested_claim(parsed_input)
            reconciliation = self.reconcile_evidence(claim, analysis, citations)
            
            final_prompt = f"""Provide a final explanation for this fact-check result:

Claim: "{claim}"
Verdict: {verdict}
Score: {score}/100

Your explanation should:
1. Clearly explain why this claim is contested, not supported or refuted
2. Reference the specific evidence that creates this contestation
3. Explain what would be needed to resolve this contestation
4. Be precise about which aspects are true and which are misleading
5. Use language that is clear, balanced, and educational

Based on the analysis:
{analysis}

And the reconciliation:
{reconciliation}

Final explanation:"""
            
            return self.call_llm(final_prompt)
            
        else:
            # For non-contested verdicts, use a simpler approach
            prompt = f"""Explain the following fact-check result:

Claim: "{claim}"
Verdict: {verdict}
Score: {score}/100
Evidence:
{chr(10).join([f"- {c}" for c in citations])}

Provide a clear, concise explanation for why this verdict was reached, referencing specific evidence.

Explanation:"""
            
            return self.call_llm(prompt)

    def reasoning_agent(self, question):
        """Main entry point for reasoning about fact-check results"""
        parsed_input = self.parse_fact_check_input(question)
        
        if not parsed_input.get('verdict'):
            # If no verdict found, use general reasoning
            return self.call_llm(question)
        
        verdict = parsed_input.get('verdict', '').lower()
        if verdict == "not enough evidence":
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
        
    def identify_statistical_pattern(self, claim, citations):
        """Identify statistical patterns in NBA data"""
        prompt = f"""Analyze the following NBA statistical claim:

Claim: "{claim}"

Evidence:
{chr(10).join([f"- {c}" for c in citations])}

Identify:
1. The player(s) mentioned
2. The specific statistics referenced (points, rebounds, assists, etc.)
3. Any seasons or teams mentioned
4. Any qualifiers or conditions (regular season, playoffs, career average, etc.)

Statistical pattern analysis:"""

        return self.call_llm(prompt)
        
    def analyze_contested_claim(self, parsed_input):
        """Override to provide NBA-specific analysis for contested claims"""
        claim = parsed_input.get('claim', '')
        citations = parsed_input.get('citation_list', [])
        
        # Check if claim involves NBA statistics
        if "NBA" in claim or any("NBA" in c for c in citations):
            statistical_analysis = self.identify_statistical_pattern(claim, citations)
            
            prompt = f"""Analyze the following contested NBA statistical claim:

Claim: "{claim}"

Evidence:
{chr(10).join([f"- {c}" for c in citations])}

Statistical Analysis:
{statistical_analysis}

Explain in detail:
1. The specific statistical discrepancies between the claim and evidence
2. If the statistics reflect different seasons, teams, or contexts
3. How career averages vs. season averages might affect interpretation
4. How different statistical qualifying criteria might apply (min. games played, etc.)
5. Whether the statistics are regular season, playoff, or combined
6. How these factors make the claim technically "contested" rather than clearly true or false

Detailed NBA statistical analysis:"""

            return self.call_llm(prompt, temperature=0.1)
        
        # Fall back to general analysis for non-NBA claims
        return super().analyze_contested_claim(parsed_input)

    def reconcile_evidence(self, claim, analysis, citations):
        """Override to provide NBA-specific reconciliation"""
        if "NBA" in claim or any("NBA" in c for c in citations):
            prompt = f"""Based on the following analysis of a contested NBA statistical claim, 
reconcile the apparently conflicting evidence:

Claim: "{claim}"

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
3. The most accurate and precise statement that could be made about this player's statistics
4. How a sports analyst would properly contextualize these numbers

NBA statistical reconciliation:"""

            return self.call_llm(prompt)
            
        # Fall back to general reconciliation for non-NBA claims
        return super().reconcile_evidence(claim, analysis, citations)


class Temporal_Context_Reasoner(EnhancedLLMReasoning):
    """
    A specialized reasoner that understands how time and context affect fact verification,
    particularly useful for claims that may have been true in one period but not another.
    """
    
    def __init__(self, llm: LLMInterface):
        super().__init__(llm)
        
    def extract_temporal_context(self, claim, citations):
        """Extract temporal context from claims and evidence"""
        prompt = f"""Extract all temporal information from this claim and evidence:

Claim: "{claim}"

Evidence:
{chr(10).join([f"- {c}" for c in citations])}

Identify:
1. All specific dates, years, seasons, or time periods mentioned
2. Any implicit temporal references
3. The chronological order of relevant events
4. Any changes over time that would affect the claim's validity

Temporal context analysis:"""

        return self.call_llm(prompt)
        
    def analyze_contested_claim(self, parsed_input):
        """Override to provide temporal-specific analysis"""
        claim = parsed_input.get('claim', '')
        citations = parsed_input.get('citation_list', [])
        
        # Extract temporal information first
        temporal_context = self.extract_temporal_context(claim, citations)
        
        prompt = f"""Analyze this temporally contested claim:

Claim: "{claim}"

Evidence:
{chr(10).join([f"- {c}" for c in citations])}

Temporal Context:
{temporal_context}

Explain in detail:
1. How different time periods affect the truth of this claim
2. Whether the claim was true during some periods and false during others
3. If there are relevant changes over time that affect the claim's validity
4. How these temporal factors make the claim "contested" rather than simply true or false

Temporal analysis:"""

        return self.call_llm(prompt, temperature=0.1)

class Multi_Perspective_Reasoner(EnhancedLLMReasoning):
    """
    A reasoner that considers multiple valid perspectives to explain
    why claims might be contested rather than simply true or false.
    """
    
    def __init__(self, llm: LLMInterface):
        super().__init__(llm)
        
    def identify_perspectives(self, claim, citations):
        """Identify different valid perspectives on the claim"""
        prompt = f"""Identify different valid perspectives on this claim:

Claim: "{claim}"

Evidence:
{chr(10).join([f"- {c}" for c in citations])}

Identify at least 2-3 different perspectives from which this claim could be evaluated,
considering differences in:
1. Definitions of key terms
2. Methodological approaches
3. Contextual assumptions
4. Frames of reference
5. Evaluation criteria

Multiple perspectives:"""

        return self.call_llm(prompt, temperature=0.2)  # More creative for perspective generation
        
    def analyze_contested_claim(self, parsed_input):
        """Override to provide multi-perspective analysis"""
        claim = parsed_input.get('claim', '')
        citations = parsed_input.get('citation_list', [])
        
        # Identify different perspectives
        perspectives = self.identify_perspectives(claim, citations)
        
        prompt = f"""Analyze this contested claim from multiple perspectives:

Claim: "{claim}"

Evidence:
{chr(10).join([f"- {c}" for c in citations])}

Perspectives:
{perspectives}

Explain in detail:
1. How different perspectives lead to different evaluations of this claim
2. Why reasonable people might disagree about this claim's validity
3. What underlying assumptions or definitions create this disagreement
4. How these multiple valid perspectives make the claim "contested" rather than simply true or false

Multi-perspective analysis:"""

        return self.call_llm(prompt, temperature=0.1)
