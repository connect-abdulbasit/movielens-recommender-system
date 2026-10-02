"""Hand-computed examples for each metric."""
import math

import pytest

from src.metrics import mae, precision_at_k, recall_at_k, rmse


def test_precision_at_k():
    # top-3 = [10, 20, 30]; relevant hits = {10, 30} -> 2/3
    assert precision_at_k([10, 20, 30, 40], {10, 30, 99}, k=3) == pytest.approx(2 / 3)


def test_precision_at_k_no_hits():
    assert precision_at_k([1, 2, 3], {9}, k=3) == 0.0


def test_recall_at_k():
    # 2 of 4 relevant items appear in top-3 -> 0.5
    assert recall_at_k([10, 20, 30, 40], {10, 30, 50, 60}, k=3) == pytest.approx(0.5)


def test_recall_at_k_empty_relevant():
    assert recall_at_k([1, 2, 3], set(), k=3) == 0.0


def test_rmse():
    # errors: 0, 1, 2 -> sqrt((0 + 1 + 4) / 3)
    assert rmse([3, 4, 5], [3, 3, 3]) == pytest.approx(math.sqrt(5 / 3))


def test_mae():
    # abs errors: 0, 1, 2 -> mean 1
    assert mae([3, 4, 5], [3, 3, 3]) == pytest.approx(1.0)
