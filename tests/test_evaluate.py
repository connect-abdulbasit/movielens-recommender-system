"""evaluate.summarise: mean and population standard deviation over folds."""
import numpy as np
import pandas as pd
import pytest

from src.evaluate import format_table, summarise


def test_summarise_mean_population_std_and_column_order():
    fold1 = pd.DataFrame({"model": ["A", "B"], "precision@10": [0.1, 0.3], "rmse": [1.0, np.nan]})
    fold2 = pd.DataFrame({"model": ["A", "B"], "precision@10": [0.3, 0.3], "rmse": [2.0, np.nan]})
    table = summarise([fold1, fold2]).set_index("model")
    assert list(table.columns) == ["precision@10_mean", "precision@10_std", "rmse_mean", "rmse_std"]
    assert table.loc["A", "precision@10_mean"] == pytest.approx(0.2)
    assert table.loc["A", "precision@10_std"] == pytest.approx(0.1)  # population std (ddof=0), not 0.1414
    assert table.loc["A", "rmse_std"] == pytest.approx(0.5)
    assert np.isnan(table.loc["B", "rmse_mean"])
    assert list(table.index) == ["A", "B"]  # model order kept


def test_format_table_shows_na_for_missing_metrics():
    fold = pd.DataFrame({"model": ["A"], "rmse": [np.nan]})
    assert format_table(summarise([fold, fold]))["rmse"].tolist() == ["n/a"]
