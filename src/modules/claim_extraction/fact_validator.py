import json
import re
from typing import List
from dataclasses import dataclass, field
from modules.claim_extraction.fact_validator_interface import *

class FactValidator(FactValidatorInterface):
    
    def _extract_json_array(self, text: str) -> list:
        """Extract JSON array from text that may have preamble/postamble"""
        try:
            # First try direct parsing
            return json.loads(text)
        except:
            # Look for array pattern
            match = re.search(r'\[(?:[^[\]]|\[[^\]]*\])*\]', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
        return None

    def validate_claim(self, claim: str, claim_type: str, passages: List[SourcePassage]) -> FactCheckResult:
        """
        The **Core Clean Interface** function.
        Returns the appropriate enforced interface type (FactCheckResult).
        """

        formatted_passages = "\n\n".join(
            [f"Passage {i+1}: {p.content}" for i, p in enumerate(passages)]
        )

        
        prompt = f"""
        You are a fact-checking assistant.

        Claim:
        "{claim}"

        Here are several passages that may support or contradict it:

        {formatted_passages}

        For each passage, classify whether it:
        - ENTAILS the claim,
        - CONTRADICTS the claim, or
        - is NEUTRAL.

        Return a list in JSON format like:
        [
        {{ "passage": 1, "label": "entailment" }},
        {{ "passage": 2, "label": "contradiction" }},
        ...
        ]
        """

        llm_response_str = self.llm.message(prompt)

        print(f"\n--- DEBUG: RAW LLM RESPONSE ---\n{llm_response_str}\n-------------------------------\n")

        # --- Start of new logic ---

        # 1. Parse LLM response
        passage_labels = self._extract_json_array(llm_response_str)
        
        if not passage_labels or not isinstance(passage_labels, list):
            # If LLM response is bad, return "Not enough evidence"
            return FactCheckResult(
                claim=claim,
                verdict="Not enough evidence",
                score=0,
                citations=[],
                features=FactCheckFeatures(0, 0, 0, 0, 0, 0)
            )

        # 2. Aggregate labels and collect passages
        entailing_passages = []
        contradicting_passages = []
        entail_scores = []
        contradict_scores = []
        all_relevance = []
        agree_domains = set()

        for label_obj in passage_labels:
            try:
                passage_index = int(label_obj.get("passage")) - 1 # 1-based to 0-based
                label = label_obj.get("label", "neutral").lower()

                if not (0 <= passage_index < len(passages)):
                    continue # Skip if passage number is out of bounds

                passage = passages[passage_index]
                all_relevance.append(passage.relevance_score)
                
                if label == "entailment":
                    entailing_passages.append(passage)
                    entail_scores.append(1.0) # Use 1.0 as score for simplicity
                    contradict_scores.append(0.0)
                    agree_domains.add(passage.domain)
                elif label == "contradiction":
                    contradicting_passages.append(passage)
                    entail_scores.append(0.0)
                    contradict_scores.append(1.0) # Use 1.0 as score
                else:
                    entail_scores.append(0.0)
                    contradict_scores.append(0.0)
            
            except (ValueError, TypeError, AttributeError):
                continue # Skip malformed entries

        # 3. Compute simplified features
        entail_scores.sort(reverse=True)
        features = FactCheckFeatures(
            entail_max=max(entail_scores) if entail_scores else 0.0,
            entail_mean3=sum(entail_scores[:3]) / 3.0 if entail_scores else 0.0,
            contradict_max=max(contradict_scores) if contradict_scores else 0.0,
            agree_domain_count=len(agree_domains),
            releliance_score_avg=sum(all_relevance) / len(all_relevance) if all_relevance else 0.0,
            recency_weight_max=0.0 # Stubbed: Calculating this requires a separate time decay function
        )

        # 4. Apply simple rules for final verdict, score, and citations
        final_verdict: VerdictType = "Not enough evidence"
        final_score = 0
        cite_passages = []

        if len(entailing_passages) > 0 and len(contradicting_passages) > 0:
            final_verdict = "Contested"
            final_score = 50 # Base score for contested claims
            # Cite one of each
            cite_passages = entailing_passages[:1] + contradicting_passages[:1]
        
        elif len(entailing_passages) > 0:
            final_verdict = "Supported"
            # Score based on number of supporting passages
            final_score = min(60 + (len(entailing_passages) * 15), 100) 
            cite_passages = entailing_passages[:3] # Cite up to 3 supporting
        
        elif len(contradicting_passages) > 0:
            final_verdict = "Refuted"
            # Score based on number of refuting passages
            final_score = min(60 + (len(contradicting_passages) * 15), 100)
            cite_passages = contradicting_passages[:3] # Cite up to 3 refuting
        
        else: # Only neutral passages found
            final_verdict = "Not enough evidence"
            final_score = 10 # Low confidence

        # 5. Format the final citation objects
        final_citations = []
        for p in cite_passages:
            final_citations.append(
                Citation(
                    url=p.url,
                    title=p.title,
                    published_at=p.published_at,
                    snippet=p.content[:250] + "..." # Use first 250 chars as snippet
                )
            )

        # 6. Return the final, structured result
        return FactCheckResult(
            claim=claim,
            verdict=final_verdict,
            score=final_score,
            citations=final_citations,
            features=features
        )