# Movie Recommender: User-Based CF vs. Naive Bayes

Course project comparing two recommenders on the MovieLens 100K dataset,
implemented from scratch with only `numpy`, `pandas` and `matplotlib`
(no recommender libraries).

## Models

- **User-based collaborative filtering** (`src/user_cf.py`): k nearest
  neighbours with Pearson or cosine similarity and optional shrinkage.
- **Naive Bayes** (`src/naive_bayes.py`): per-user model over movie genres.
  Estimates P(Like), P(genre | Like) and P(genre | Dislike) with Laplace
  smoothing and ranks movies by P(Like | genres).
- **Baselines** (`src/baselines.py`): random and popularity.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dataset lives in `data/ml-100k/` (official `u1..u5` train/test splits).

## Usage

```bash
python main.py recommend --user 196 --method cf    # or --method nb
python main.py evaluate                            # writes results/results.csv
pytest                                             # run metric tests
```

User and movie ids are 1-based (as in MovieLens) at every public interface
and converted to 0-based indices internally.

## Evaluation protocol

Official 5-fold splits `u1..u5`. A test item is *relevant* if its rating is
>= 4. Only movies the user did not rate in the training set are ranked.
Metrics: Precision@K, Recall@K, RMSE, MAE.

## Results

_TODO: fill in after running `python main.py evaluate`._

| Model | Precision@10 | Recall@10 | RMSE | MAE |
|-------|--------------|-----------|------|-----|
| Random | | | | |
| Popularity | | | | |
| User-CF | | | | |
| Naive Bayes | | | | |

## Layout

```
src/        data loading, models, metrics, explanations, evaluation
tests/      unit tests
results/    generated results (CSV committed, PNGs ignored)
report/     final report
proposal/   project proposal
```
