import importlib.util
import re
from pathlib import Path

import pandas as pd
import pytest

from src.human_eval import LIST_LABELS, MODELS, write_kit

TABULATE_PATH = Path(__file__).resolve().parent.parent / "human_eval" / "tabulate.py"
FAST_NN = {"epochs": 2}

def _tabulate_module():
    spec = importlib.util.spec_from_file_location("human_eval_tabulate", TABULATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.fixture(scope="module")
def kit_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("human_eval")
    write_kit(out_dir=out, n_users=3, seed=5, nn_params=FAST_NN)
    return out

def _sheet_tables(sheet):
    tables, user = {}, None
    for line in sheet.split("## Part 2")[0].splitlines():
        heading = re.match(r"### User \d+ of \d+ \(MovieLens user (\d+)\)", line)
        if heading:
            user = int(heading.group(1))
            tables[user] = {label: [] for label in LIST_LABELS}
        elif re.match(r"\| \d+ \|", line):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            for label, title in zip(LIST_LABELS, cells[1:]):
                tables[user][label].append(title)
    return tables

def test_sheet_never_names_a_model(kit_dir):
    sheet = (kit_dir / "rater_sheet.md").read_text(encoding="utf-8")
    from src.data_loader import load_movies

    for title in load_movies()["title"]:
        sheet = sheet.replace(title, "")
    for word in ["neural", "network", "naive", "bayes", "popularity", "popular", "nn", "nb"]:
        assert not re.search(rf"\b{word}\b", sheet, flags=re.IGNORECASE), word
    assert "(score" not in sheet

def test_answer_key_maps_labels_back_to_the_right_model(kit_dir):
    from src.baselines import PopularityRecommender
    from src.data_loader import build_matrix, load_movies, load_ratings

    sheet = (kit_dir / "rater_sheet.md").read_text(encoding="utf-8")
    key = pd.read_csv(kit_dir / "answer_key.csv")
    tables = _sheet_tables(sheet)
    titles = load_movies().set_index("item_id")["title"]
    popularity = PopularityRecommender().fit(build_matrix(load_ratings(1)[0]))
    lists = key[key["part"] == "list"]
    assert len(tables) == 3
    for user, labels in tables.items():
        mapping = lists[lists["user_id"] == user].set_index("label")["model"]
        assert sorted(mapping) == sorted(MODELS)
        pop_label = mapping[mapping == "Popularity"].index[0]
        assert labels[pop_label] == [titles[m] for m, _ in popularity.recommend(user, 10)]
    explanations = key[key["part"] == "explanation"]
    assert sorted(explanations["model"].unique()) == ["Naive Bayes", "Neural Network"]
    assert all(sorted(labels) == ["X", "Y"] for labels in explanations.groupby("user_id")["label"].apply(list))

def test_tabulate_refuses_the_empty_template(kit_dir, capsys):
    module = _tabulate_module()
    status = module.main([str(kit_dir / "ratings_template.csv"), str(kit_dir / "answer_key.csv")])
    assert status == 1
    assert "empty ratings_template.csv" in capsys.readouterr().out

def test_tabulate_maps_through_the_key(kit_dir, tmp_path):
    module = _tabulate_module()
    template = pd.read_csv(kit_dir / "ratings_template.csv", dtype=str, keep_default_na=False)
    key = pd.read_csv(kit_dir / "answer_key.csv", dtype=str, keep_default_na=False)
    lists = key[key["part"] == "list"].set_index(["user_id", "label"])["model"]
    reasons = key[key["part"] == "explanation"].set_index(["user_id", "label"])["model"]
    filled = []
    for rater in ("r1", "r2"):
        sheet = template.copy()
        sheet["rater"] = rater
        sheet["fit_score"] = [("5" if lists[(u, l)] == "Neural Network" else "2") for u, l in zip(sheet["user_id"], sheet["list_label"])]
        for user in sheet["user_id"].unique():
            first = sheet.index[(sheet["user_id"] == user) & (sheet["list_label"] == "A")][0]
            nn_label = next(l for l in "ABC" if lists[(user, l)] == "Neural Network")
            sheet.loc[first, "best_list"] = nn_label
            for label in "XY":
                sheet.loc[first, f"clarity_{label}"] = "4" if reasons[(user, label)] == "Neural Network" else "3"
        filled.append(sheet)
    path = tmp_path / "ratings.csv"
    pd.concat(filled).to_csv(path, index=False)
    result = module.tabulate(module.load(path), module.load(kit_dir / "answer_key.csv"))
    assert result["raters"] == 2
    assert result["fit"].loc["Neural Network", "mean"] == 5 and result["fit"].loc["Popularity", "mean"] == 2
    assert result["wins"].to_dict() == {"Neural Network": 6}
    assert result["clarity"].loc["Neural Network", "mean"] == 4 and result["clarity"].loc["Naive Bayes", "mean"] == 3
