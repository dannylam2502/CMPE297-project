"""
input_extractor.py

Claim extraction module using OpenAI API.
Extracts structured, verifiable claims from user input.
"""

import os
import json
import re
import modules.input_extraction.input_normalizer as input_normalizer

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
From the given INPUT TEXT, extract the **atomic factual claim** that is suitable for fact checking.
This will be a single cleaned but semantically faithful claim.
Normalization means making the text syntactically clear and standalone, not correcting, negating, or fact-checking it.
If the input is false, biased, or implausible, you must preserve it exactly as asserted — do not rewrite it to be true.

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
- Do NOT correct factual errors or flip the truth value of the claim.
  Example: If input says "Mexico is in Canada", the normalized claim must keep the same meaning ("Mexico is located in Canada"), NOT "Mexico is not in Canada".
- Normalization is limited to grammar, casing, or removing fillers.
  Do not introduce negation, modality, or correction unless they already exist in the input.
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

def extract_claim_from_input(
    llm: LLMInterface,
    user_input: str,
    *,
    preclean: bool | None = None,
    tz: str = "America/Los_Angeles",
) -> dict:
    """
    Extract structured claim from user input.

    Args:
        user_input: Raw user text (expected single claim)
        preclean:  If True, apply OCR/ASR normalization before prompting.
                   If None, read from env PRE_CLEAN (default False).
        tz:        Timezone hint for downstream normalization (if you later add it to the prompt)

    Returns:
        Dict matching your schema (with 'original_input' attached)
    """
    # Decide pre-cleaning via param or env
    if preclean is None:
        preclean = os.getenv("PRE_CLEAN", "0").strip() in {"1", "true", "True", "yes", "Y"}

    original_input = user_input
    cleaned_input = input_normalizer.normalize_ocr_asr(user_input) if preclean else user_input

    response_text = call_to_structure(llm, cleaned_input)
    structured = extract_json_from_text(response_text)

    if structured is None:
        # Fallback minimal structure
        return {
            "doc_meta": {
                "language": "en",
                "source_type": "post",
                "extraction_quality_note": "LLM JSON parse failed; fallback.",
            },
            "claims": [{
                "id": "C1",
                "text_span": original_input,
                "normalized": cleaned_input,
                "type": "unknown",
                "topic": "other",
                "temporal": {"when_text": None, "when_iso": None},
                "quantity": {"value_text": None, "value_num": None, "unit": None},
                "stance": "uncertain",
                "modality_hedges": [],
                "evidence_cues": {"urls": [], "quoted_sources": [], "media_mentions": [], "numbers_in_text": []},
                "sensitivity": {"domain": ["other"], "harm_risk": "low"},
                "verifiability": {"is_checkable": False, "best_evidence_types": []},
                "attribution": {"speaker": None, "speaker_type": None},
                "context": {"surrounding_sentence": None, "thread_relation": "original"}
            }],
            "non_claim_spans": [],
            "original_input": original_input
        }

    # Attach helpful metadata & provenance
    structured.setdefault("doc_meta", {})
    note_bits = []
    if preclean:
        note_bits.append("preclean: on")
    if note_bits:
        prev = structured["doc_meta"].get("extraction_quality_note", "")
        structured["doc_meta"]["extraction_quality_note"] = (prev + ("; " if prev else "") + ", ".join(note_bits))

    structured.setdefault("original_input", original_input)
    return structured