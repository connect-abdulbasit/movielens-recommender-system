# Movie Recommender: Neural Network vs. Naive Bayes

Course project comparing two recommenders on the MovieLens 100K dataset,
implemented from scratch with only `numpy`, `pandas` and `matplotlib`
(no recommender libraries).

## Models

- **Neural network** (`src/features.py`, `src/neural_net.py`,
  `src/neural_recommender.py`): a small MLP written in pure NumPy that predicts
  how far a user's rating of a movie is from their own average, from
  interpretable features (genres, era, user level, per-genre taste, movie
  statistics). Predictions are explained by occlusion. See
  [docs/how_it_works.md](docs/how_it_works.md).
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

For this handoff, copy the original project’s entire data/ml-100k folder into the extracted project, beside src/. On Windows activate the environment with .venv\Scripts\activate. Python 3.10+ is required.

The dataset lives in `data/ml-100k/` (official `u1..u5` train/test splits).

## Usage

```bash
python main.py recommend --user 196 --method nn    # or --method nb
python main.py evaluate                            # writes results/results.csv
pytest                                             # run all tests
```

User and movie ids are 1-based (as in MovieLens) at every public interface
and converted to 0-based indices internally.

## Evaluation protocol

Official 5-fold splits `u1..u5`. A test item is *relevant* if its rating is
>= 4. Only movies the user did not rate in the training set are ranked.
Metrics: Precision@K, Recall@K, RMSE, MAE.

## Results

Five-fold mean ± population standard deviation. The two ranking columns are fractions.

| Model | Precision@10 | Recall@10 | RMSE | MAE |
|---|---:|---:|---:|---:|
| Random | 0.0099 ± 0.0032 | 0.0056 ± 0.0004 | 1.1256 ± 0.0156 | 0.9447 ± 0.0131 |
| Popularity | 0.1567 ± 0.0330 | 0.1360 ± 0.0100 | 1.0246 ± 0.0063 | 0.8174 ± 0.0061 |
| Naive Bayes | 0.0366 ± 0.0103 | 0.0242 ± 0.0034 | N/A | N/A |
| Neural Network | 0.0941 ± 0.0316 | 0.0570 ± 0.0099 | 0.9401 ± 0.0057 | 0.7447 ± 0.0047 |

## Layout

```
src/        data loading, features, models, metrics, explanations, evaluation
docs/       how_it_works.md: plain-language explanation of the neural model
tests/      unit tests
results/    generated results (CSV committed, PNGs ignored)
report/     final report
proposal/   project proposal
```
