import pandas as pd
import pytest

from main import main
from src.export import COLUMNS, ranking_table

class FixedModel:

    def recommend(self, user, n=10):
        return [(3, 0.9), (1, 0.5), (2, 0.1)][:n]

TITLES = pd.Series({1: "A", 2: "B", 3: "C"})
TEST = pd.DataFrame({"user_id": [1, 2], "item_id": [3, 2], "rating": [5, 2]})

def test_table_has_one_row_per_user_and_rank():
    table = ranking_table(FixedModel(), [1, 2], 3, TITLES, TEST)
    assert list(table.columns) == COLUMNS
    assert len(table) == 6
    assert table[table.user_id == 1]["rank"].tolist() == [1, 2, 3]
    assert table[table.user_id == 1]["title"].tolist() == ["C", "A", "B"]

def test_test_rating_is_joined_only_where_held_out():
    table = ranking_table(FixedModel(), [1, 2], 3, TITLES, TEST).set_index(["user_id", "movie_id"])
    assert table.loc[(1, 3), "test_rating"] == 5
    assert table.loc[(2, 2), "test_rating"] == 2
    assert pd.isna(table.loc[(1, 1), "test_rating"])

def test_top_limits_list_length():
    assert len(ranking_table(FixedModel(), [1, 2], 1, TITLES, TEST)) == 2

@pytest.mark.parametrize("top", ["0", "-2"])
def test_top_must_be_positive(top):
    with pytest.raises(SystemExit, match="--top must be positive"):
        main(["export", "--top", top])
