"""
input_extractor.py

Claim extraction module using OpenAI API.
Extracts structured, verifiable claims from user input.
"""

import os
import json
import re

from modules.llm.llm_engine_interface import LLMInterface

try:
    import openai
except ImportError:
    raise ImportError("Install openai package: pip install openai")

from dotenv import load_dotenv

MODEL_NAME = "gpt-4o-mini"

SCHEMA_INSTRUCTIONS = """
You are ClaimExtractor, a careful NLP tool that extracts clean, verifiable claims from messy text to combat misinformation. 

## Task
From the given INPUT TEXT, extract the **atomic factual claim** that is suitable for fact checking. This will be a single cleaned claim, normalized for verifiability and proper attribution.

### What counts as a "claim"?
- A claim must be an **assertion** or **fact** that can be verified as true/false or falsifiable (e.g., numerical, comparative, causational).
- You must **exclude** opinions, perspectives, or rhetorical questions unless they contain checkable predicates.
- Claims should be **context-independent** (e.g., causal statements, numerical data, temporal assertions).

### Output Format
Return a **strict JSON** that matches the SCHEMA below. Do not add commentary.

### SCHEMA
{
  "doc_meta": {
    "language": "en",
    "source_type": "string",
    "extraction_quality_note": "string"
  },
  "claims": [
    {
      "id": "C1",
      "text_span": "string",
      "normalized": "string",
      "type": "string",
      "topic": "string",
      "subject_entities": [ {"name":"string","type":"string"} ],
      "objects_entities":  [ {"name":"string","type":"string"} ],
      "temporal": {
        "when_text": "string|null",
        "when_iso": "string|null"
      },
      "location": "string|null",
      "quantity": {
        "value_text": "string|null",
        "value_num": "number|null",
        "unit": "string|null"
      },
      "stance": "string",
      "modality_hedges": ["string"],
      "evidence_cues": {
        "urls": ["string"],
        "quoted_sources": ["string"],
        "media_mentions": ["string"],
        "numbers_in_text": ["string"]
      },
      "sensitivity": {
        "domain": ["string"],
        "harm_risk": "low|medium|high"
      },
      "verifiability": {
        "is_checkable": true,
        "best_evidence_types": ["string"]
      },
      "attribution": {
        "speaker": "string|null",
        "speaker_type": "string|null"
      },
      "context": {
        "surrounding_sentence": "string|null",
        "thread_relation": "original|reply|quote|reshare|unknown"
      }
    }
  ],
  "non_claim_spans": ["string"]
}

## Rules
- Ensure that the extracted claim is **atomic** and properly normalized for checkability.
- Normalize temporal expressions (e.g., convert "next week" to "YYYY-MM-DD").
- If the input is noisy (OCR/ASR errors), make reasonable guesses but mark uncertain claims with **stance="uncertain"**.
- Keep **modality_hedges** (like "may" or "could") in the output if the claim is uncertain.
- Only return **one cleaned claim** in the output (`claims[0]`).
"""

def call_to_structure(llm: LLMInterface, text: str) -> str:
    """Call OpenAI API to extract structured claim"""
    system_msg = (
        "You are a strict JSON formatter. Convert user text into the JSON schema provided. "
        "Do not add any extra fields or commentary."
    )
    user_msg = f"{SCHEMA_INSTRUCTIONS}\n\nUser input:\n\"\"\"\n{text}\n\"\"\""
    
    try:
        resp = llm.raw_messages([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ])
        resp = openai.responses.create(
            model=MODEL_NAME,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {e}")
    
    return resp

def extract_json_from_text(text: str) -> dict:
    """Extract JSON from potentially markdown-wrapped response"""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"(\{(?:.|\n)*\})", text)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    return None

def extract_claim_from_input(llm: LLMInterface, user_input: str) -> dict:
    """
    Extract structured claim from user input.
    
    Args:
        user_input: Raw user query
        
    Returns:
        Dict with Danny's full schema structure
    """
    response_text = call_to_structure(llm, user_input)
    structured = extract_json_from_text(response_text)
    
    if structured is None:
        # Fallback
        return {
            "claims": [{
                "id": "C1",
                "normalized": user_input,
                "type": "unknown"
            }],
            "original_input": user_input
        }
    
    structured.setdefault("original_input", user_input)
    return structured