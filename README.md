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

## Change from the proposal

The original proposal used User-Based Collaborative Filtering as the second
technique. With the teacher's approval it was replaced by the neural network
recommender; the revised proposal in `proposal/` describes the final pair
(Neural Network vs. Naive Bayes).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows activate the environment with `.venv\Scripts\activate`.
Python 3.9 or later is required (tested on 3.9.6).

The dataset is included in the repository in `data/ml-100k/` (official
`u1..u5` train/test splits).

## Usage

```bash
python main.py recommend --user 196 --method nn    # or --method nb
python main.py evaluate                            # writes results/results.csv and fold_results.csv
python main.py plots                               # figures in results/figures/
python main.py export                              # every model's ranking for every user (see below)
python main.py human-eval                          # blind human-evaluation kit in human_eval/
python scripts/check_explanations.py               # checks NN explanations never misstate a taste
pytest                                             # run all tests
```

- `plots` draws `fig1_ranking.png`, `fig2_rating_error.png`, and
  `fig4_training_curve.png`. For the last one it trains the default network
  on fold 1 and saves its per-epoch losses to `results/training_curve.csv`.
- `export` trains all four models on one fold (`--fold`, default 1) and
  writes `results/recommendations/fold1/<model>.csv`: for every user, every
  movie they did not rate in training, ranked, with columns `user_id, rank,
  movie_id, title, score, test_rating`. `score` is predicted stars (NN),
  P(like) (NB), number of ratings (Popularity) or the constant global mean
  (Random). `test_rating` is the user's hidden rating, empty if none; 4 or
  more counts as a hit. `--top 10` keeps only each user's top 10. The full
  files are large and are not committed (see `.gitignore`).
- `human-eval` writes a printable rater sheet, an empty ratings template and
  an answer key; see [human_eval/README.md](human_eval/README.md). The
  collected ratings are in `human_eval/ratings.csv`.

User and movie ids are 1-based (as in MovieLens) at every public interface
and converted to 0-based indices internally.

## Evaluation protocol

Official 5-fold splits `u1..u5`. A test item is *relevant* if its rating is
>= 4. Only movies the user did not rate in the training set are ranked.
Metrics: Precision@K, Recall@K, RMSE, MAE.

## Results

Five-fold mean ± population standard deviation. The two ranking columns are fractions.
Produced by `python main.py evaluate` (`results/results.csv`).

| Model | Precision@10 | Recall@10 | RMSE | MAE |
|---|---:|---:|---:|---:|
| Random | 0.0099 ± 0.0032 | 0.0056 ± 0.0004 | 1.1256 ± 0.0156 | 0.9447 ± 0.0131 |
| Popularity | 0.1567 ± 0.0330 | 0.1360 ± 0.0100 | 1.0246 ± 0.0063 | 0.8174 ± 0.0061 |
| Naive Bayes | 0.0366 ± 0.0103 | 0.0242 ± 0.0034 | N/A | N/A |
| Neural Network | 0.0941 ± 0.0316 | 0.0570 ± 0.0099 | 0.9401 ± 0.0057 | 0.7447 ± 0.0047 |

## Layout

```
main.py         command-line interface (recommend, evaluate, plots, export, human-eval)
src/            data loading, features, models, metrics, explanations, evaluation,
                plots, human-evaluation kit
scripts/        check_explanations.py: explanation-faithfulness check
tests/          unit tests
docs/           how_it_works.md: plain-language explanation of the neural model
results/        result CSVs; figures/ holds the report figures (tracked)
human_eval/     blind human-evaluation kit (rater sheet, template, answer key, tabulate.py)
data/ml-100k/   MovieLens 100K
report/         final report
proposal/       revised project proposal
```
