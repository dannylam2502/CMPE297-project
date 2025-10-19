# tests/test_simple_factchecker.py
import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from modules.claim_extraction.fact_validator_interface import (
    SourcePassage,
    ClaimCheckResult,
    Citation
)

from modules.claim_extraction.fact_validator import (
    FactValidator
)

# Minimal LLM stub to satisfy the constructor of SimpleFactChecker -> FactChecker
class DummyLLM:
    pass


@pytest.fixture
def checker() -> FactValidator:
    return FactValidator(DummyLLM())


# --------------------
# Helper method tests
# --------------------

def test_sigmoid_basic(checker: FactValidator):
    # Known value at 0
    assert checker._sigmoid(0.0) == pytest.approx(0.5)

    # Monotonic behavior: larger x -> larger sigmoid(x)
    xs = [-4, -2, -1, 0, 1, 2, 4]
    vals = [checker._sigmoid(x) for x in xs]
    assert all(0.0 < v < 1.0 for v in vals)
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    # Symmetry: s(x) = 1 - s(-x)
    for x in [0.3, 1.7, 5.2]:
        assert checker._sigmoid(x) == pytest.approx(1.0 - checker._sigmoid(-x))


def test_calculate_recency_weight_boundaries(checker: FactValidator):
    now = datetime.now()

    # <= 30 days -> 1.0
    assert checker._calculate_recency_weight(now) == pytest.approx(1.0)
    assert checker._calculate_recency_weight(now - timedelta(days=30)) == pytest.approx(1.0)

    # >= 365 days -> 0.0
    assert checker._calculate_recency_weight(now - timedelta(days=365)) == pytest.approx(0.0)
    assert checker._calculate_recency_weight(now - timedelta(days=800)) == pytest.approx(0.0)

    # Linear decay between 30 and 365
    w_31 = checker._calculate_recency_weight(now - timedelta(days=31))
    w_mid = checker._calculate_recency_weight(now - timedelta(days=(30 + 365) // 2))
    w_364 = checker._calculate_recency_weight(now - timedelta(days=364))
    assert 0.0 < w_31 < 1.0
    assert 0.0 < w_mid < w_31
    assert 0.0 < w_364 < w_mid
