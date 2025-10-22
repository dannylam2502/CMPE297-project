# tests/test_simple_factchecker.py
import math
from datetime import datetime, timedelta

from modules.llm.llm_ollama import (llm_ollama)

import numpy as np
import pytest

from modules.claim_extraction.fact_validator_interface import *

from modules.claim_extraction.fact_validator import (
    FactValidator
)

# Minimal LLM stub to satisfy the constructor of SimpleFactChecker -> FactChecker
class DummyLLM:
    pass


@pytest.fixture
def checker() -> FactValidator:
    llm = llm_ollama()
    return FactValidator(llm)


# --------------------
# Helper method tests
# --------------------

def test_simple_print_check(checker: FactValidator):
    # 1. Define passages with blank metadata as requested
    passages_list = [
        SourcePassage(
            content="A recent study confirms that the sky is blue due to Rayleigh scattering.",
            relevance_score=0.95,
            url="",  # Blank
            domain="", # Blank
            title="", # Blank
            published_at=datetime.now()
        ),
        SourcePassage(
            content="Atmospheric particles scatter blue light more than other colors, making the sky appear blue.",
            relevance_score=0.90,
            url="", # Blank
            domain="", # Blank
            title="", # Blank
            published_at=datetime.now()
        ),
         SourcePassage(
            content="The color of the sky on Mars is reddish-pink.",
            relevance_score=0.70,
            url="", # Blank
            domain="", # Blank
            title="", # Blank
            published_at=datetime.now()
        )
    ]
    
    # 2. Define the claim to check
    user_input = "The sky is blue."
    claim_type = "factual"

    # 3. Call the function and print the result
    print("\n--- Running Simple Test ---")
    result = checker.validate_claim(user_input, claim_type, passages_list)
    print(result) # This will use the __str__ method you added
    print("---------------------------\n")

    # A minimal check for a test runner
    assert result is not None
    assert result.claim == user_input