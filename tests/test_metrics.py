import math

import pytest

from src.metrics import mae, precision_at_k, recall_at_k, rmse

def test_precision_at_k():
    assert precision_at_k([10, 20, 30, 40], {10, 30, 99}, k=3) == pytest.approx(2 / 3)

def test_precision_at_k_no_hits():
    assert precision_at_k([1, 2, 3], {9}, k=3) == 0.0

def test_recall_at_k():
    assert recall_at_k([10, 20, 30, 40], {10, 30, 50, 60}, k=3) == pytest.approx(0.5)

def test_recall_at_k_empty_relevant():
    assert recall_at_k([1, 2, 3], set(), k=3) == 0.0

def test_rmse():
    assert rmse([3, 4, 5], [3, 3, 3]) == pytest.approx(math.sqrt(5 / 3))

def test_mae():
    assert mae([3, 4, 5], [3, 3, 3]) == pytest.approx(1.0)
