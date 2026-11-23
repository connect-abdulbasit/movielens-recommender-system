"""experiments.py on a tiny synthetic data set (no MovieLens files, a few seconds)."""
import numpy as np
import pandas as pd
import pytest

from src import experiments
from src.experiments import run_experiments, setting


@pytest.fixture
def fake_fold(toy_data):
    """A data_for_fold function: 20% of the toy ratings become the test set."""
    matrix, features, names = toy_data

    def data_for_fold(fold):
        rng = np.random.default_rng(fold)
        users, items = np.nonzero(matrix)
        test_mask = rng.random(len(users)) < 0.2
        train = matrix.copy()
        train[users[test_mask], items[test_mask]] = 0
        test = pd.DataFrame({"user_id": users[test_mask] + 1, "item_id": items[test_mask] + 1,
                             "rating": matrix[users[test_mask], items[test_mask]]})
        return train, test, features, names

    return data_for_fold


SETTINGS = [
    setting("NN size", "tiny (default here)", "nn", default=True, epochs=2, hidden=(4,)),
    setting("NN size", "same config again", "nn", epochs=2, hidden=(4,)),
    setting("NB alpha", "alpha 1", "nb", default=True),
    setting("Reference", "Popularity", "popularity", default=True),
    setting("Reference", "Random", "random", default=True),
]


def test_tiny_run_writes_both_csvs_with_population_std(fake_fold, tmp_path):
    summary = run_experiments(SETTINGS, folds=[1, 2], data_for_fold=fake_fold, out_dir=tmp_path, resume=False)
    by_fold = pd.read_csv(tmp_path / "experiments_by_fold.csv")
    assert len(by_fold) == len(SETTINGS) * 2
    assert len(summary) == len(SETTINGS)
    assert pd.read_csv(tmp_path / "experiments.csv").shape == summary.shape
    nb = summary[summary["model"] == "nb"].iloc[0]
    assert np.isnan(nb["rmse_mean"])  # Naive Bayes predicts no star ratings
    pop = by_fold[by_fold["model"] == "popularity"]["precision@10"]
    assert summary[summary["model"] == "popularity"]["precision@10_std"].iloc[0] == pytest.approx(pop.std(ddof=0))
    nn = summary[summary["model"] == "nn"]
    assert nn["precision@10_mean"].nunique() == 1  # same config -> same numbers


def test_identical_configs_run_once_and_resume_uses_the_cache(fake_fold, tmp_path, capsys):
    run_experiments(SETTINGS, folds=[1, 2], data_for_fold=fake_fold, out_dir=tmp_path, resume=False)
    first = capsys.readouterr().out
    assert first.count("nn epochs=2 hidden=(4,)") == 2  # once per fold, not once per setting
    again = run_experiments(SETTINGS, folds=[1, 2], data_for_fold=fake_fold, out_dir=tmp_path, resume=True)
    out = capsys.readouterr().out
    assert "(0 computed, 8 cached)" in out
    pd.testing.assert_frame_equal(again, pd.read_csv(tmp_path / "experiments.csv"), check_dtype=False)


def test_default_settings_are_not_changed_by_the_experiment_list():
    defaults = [s for s in experiments.SETTINGS if s.default and s.model in ("nn", "nb")]
    assert defaults and all(s.params == () for s in defaults)  # defaults = model defaults, nothing overridden
