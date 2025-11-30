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


def classify_input_category(text: str) -> str:
    """
    Classify the original user input as:
      - "question": primarily interrogative (e.g., ends with '?', WH/aux start)
      - "claim": a declarative statement likely containing an assertion
      - "other": empty, pure opinion, or unclear

    This is a lightweight heuristic used to set the top-level `input_category`
    field, independent of the LLM.
    """
    if not text:
        return "other"

    s = text.strip()
    if not s:
        return "other"

    lower = s.lower()

    # If it clearly looks like a question
    if s.endswith("?"):
        return "question"

    # Leading question phrases / auxiliaries
    question_starts = (
        "who", "what", "when", "where", "why", "how",
        "is", "are", "was", "were",
        "do", "does", "did",
        "can", "could", "should", "would", "will",
        "have", "has", "had",
        "am",
        "is it true that", "do you think", "could it be that",
    )
    if lower.startswith(question_starts):
        return "question"

    # Otherwise treat as a (potential) claim/statement by default
    return "claim"


SCHEMA_INSTRUCTIONS = """
You are ClaimExtractor, a careful NLP tool that extracts clean, verifiable claims from messy text to combat misinformation. 

## Task
From the given INPUT TEXT, extract the **atomic factual claim** that is suitable for fact checking.
This will be a single cleaned but semantically faithful claim.
Normalization means making the text syntactically clear and standalone, not correcting, negating, or fact-checking it.
If the input is false, biased, or implausible, you must preserve it exactly as asserted — do not rewrite it to be true.

You must also determine whether the original INPUT TEXT is primarily a **question**, a **claim**, or **other**:
- Use "question" if the user is asking whether a factual proposition is true or false (e.g., "Is it true that vaccines cause autism?").
- Use "claim" if the user is directly asserting a proposition.
- Use "other" if it does not primarily express a factual question or assertion.

Even if the input is phrased as a question, you must still extract the underlying **proposition** as a normalized claim where possible.
Example:
- Input: "Is it true that Mexico is in Canada?"
  - normalized claim should be: "Mexico is located in Canada."

### What counts as a "claim"?
- A claim must be an **assertion** or **fact** that can be verified as true/false or falsifiable (e.g., numerical, comparative, causational).
- You must **exclude** opinions, perspectives, or rhetorical questions unless they contain checkable predicates.
- Claims should be **context-independent** (e.g., causal statements, numerical data, temporal assertions).

### Output Format
Return a **strict JSON** that matches the SCHEMA below. Do not add commentary.
Do not add fields that are not listed in the schema.

### SCHEMA
{
  "input_category": "question|claim|other",
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
- Normalize temporal expressions (e.g., convert "next week" to "YYYY-MM-DD") when possible.
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
        "Return only valid JSON. Do not add any extra fields beyond the schema and do not add commentary."
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
        user_input: Raw user text (expected single claim or question about a claim)
        preclean:  If True, apply OCR/ASR normalization before prompting.
                   If None, read from env PRE_CLEAN (default False).
        tz:        Timezone hint for downstream normalization (if you later add it to the prompt)

    Returns:
        Dict matching your schema (with 'original_input' attached and a top-level 'input_category').
    """
    # Decide pre-cleaning via param or env
    if preclean is None:
        preclean = os.getenv("PRE_CLEAN", "0").strip() in {"1", "true", "True", "yes", "Y"}

    original_input = user_input
    cleaned_input = input_normalizer.normalize_ocr_asr(user_input) if preclean else user_input

    # Local heuristic classification (robust even if LLM doesn't follow schema perfectly)
    input_category = classify_input_category(original_input)

    response_text = call_to_structure(llm, cleaned_input)
    structured = extract_json_from_text(response_text)

    if structured is None:
        # Fallback minimal structure
        return {
            "input_category": input_category,
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
                "subject_entities": [],
                "objects_entities": [],
                "temporal": {"when_text": None, "when_iso": None},
                "location": None,
                "quantity": {"value_text": None, "value_num": None, "unit": None},
                "stance": "uncertain",
                "modality_hedges": [],
                "evidence_cues": {
                    "urls": [],
                    "quoted_sources": [],
                    "media_mentions": [],
                    "numbers_in_text": [],
                },
                "sensitivity": {"domain": ["other"], "harm_risk": "low"},
                "verifiability": {"is_checkable": False, "best_evidence_types": []},
                "attribution": {"speaker": None, "speaker_type": None},
                "context": {"surrounding_sentence": None, "thread_relation": "original"},
            }],
            "non_claim_spans": [],
            "original_input": original_input,
        }

    # Attach helpful metadata & provenance
    structured.setdefault("doc_meta", {})
    note_bits = []
    if preclean:
        note_bits.append("preclean: on")
    if note_bits:
        prev = structured["doc_meta"].get("extraction_quality_note", "")
        structured["doc_meta"]["extraction_quality_note"] = (
            prev + ("; " if prev else "") + ", ".join(note_bits)
        )

    # Ensure input_category is present and aligned with our local classification
    # (LLM may have followed schema; if not, we overwrite/insert.)
    structured["input_category"] = input_category

    structured.setdefault("original_input", original_input)
    return structured
