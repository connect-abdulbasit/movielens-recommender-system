# How the neural recommender works

This page explains `src/features.py`, `src/neural_net.py` and
`src/neural_recommender.py` in plain language. Only NumPy is used: the network,
its training and the explanations are written from scratch.

## The idea in one sentence

For a (user, movie) pair we compute a few dozen understandable numbers
("features"), feed them to a very small neural network, and the network says
*how far above or below the user's own average rating* this movie will be.

```
prediction = user's mean rating + network(features)      (clipped to 1..5)
```

Working with "difference from the user's mean" removes the easy part (some
people rate everything high) so the network can focus on taste.

## Step 1: features (`src/features.py`)

All statistics come from the **training** ratings only.

| Group | Features | Meaning |
|-------|----------|---------|
| Movie | 19 genre flags, 9 release-era flags | What kind of movie, and when it was made |
| User | mean rating - 3.5, log(1 + #ratings) / 5 | Generous or harsh rater, and how active |
| Taste | one number per genre of the movie, plus their average | Does this user rate this genre above their own average? |
| Movie stats (optional) | log(1 + #ratings) / 6, shrunk average rating | How popular and how well liked the movie is |

### Worked example

Ana rated four movies: three Sci-Fi movies with 5, 5, 5 and one Drama with 1.

* Her mean rating is (5 + 5 + 5 + 1) / 4 = **4.0**.
* Her *taste* for Sci-Fi is `(sum + m * mean) / (count + m) - mean` with `m = 5`:
  `(15 + 5 * 4.0) / (3 + 5) - 4.0 = 0.375`. Positive: she likes Sci-Fi more
  than her average. The `m = 5` "virtual ratings at her mean" stop a taste
  based on only one or two ratings from looking extreme (this is called
  *shrinkage*). `tests/test_features.py` checks this exact number.
* A Sci-Fi movie that three people rated, 4.5 on average, with a global mean of
  3.53, gets a shrunk average of `(13.5 + 5 * 3.53) / (3 + 5) - 3.53 = 0.36`
  instead of a trusted-too-much 0.97.

For a Sci-Fi movie Ana has not seen, the feature vector contains the Sci-Fi
flag, her Sci-Fi taste (+0.375), her level (4.0 - 3.5), the movie's
statistics, and so on. Genres the movie does not have get taste 0.

### Why leave-one-out?

Training pairs are ratings we already know. If Ana's rating of movie X is
included in "Ana's mean", "Ana's taste for Sci-Fi" and "X's average rating",
the features secretly contain the answer and the network would learn to copy
it. It would then look great in training and fail on new movies.

So for a training pair we first **remove that one rating** from the user's
statistics, the user's genre statistics, the movie's statistics and the global
mean, then compute the features. At prediction time the movie is not in the
training data, so nothing needs removing. `tests/test_features.py` proves it:
features with leave-one-out are identical to the features computed from a matrix
in which that rating was deleted.

## Step 2: the network (`src/neural_net.py`)

A multi-layer perceptron with layers `[n_features, 32, 16, 1]`:

* each layer multiplies by a weight matrix, adds a bias, and (for the hidden
  layers) applies **ReLU** (`max(0, x)`) so the model can be non-linear;
* the output layer is linear because the answer can be any real number;
* weights start with **He initialisation** (random numbers with variance
  `2 / n_inputs`) so signals neither vanish nor explode through ReLU layers.

## Step 3: training (`src/neural_recommender.py`)

* **Target**: `rating - user mean` (leave-one-out user mean).
* **Features** are standardised (subtract the training mean, divide by the
  training spread) so that all of them have a similar scale.
* **Loss**: mean squared error, plus a small **L2 penalty** on the weights
  (`l2 * sum(weights^2)`) which keeps weights small and reduces overfitting.
* **Backpropagation** (`MLP.gradients`) applies the chain rule layer by layer,
  from the output back to the input, to get the gradient of the loss with respect to
  every weight. `tests/test_neural_net.py` compares it with numerical
  finite-difference gradients.
* **Adam** (`MLP.step`) is gradient descent with two improvements: it follows a
  smoothed average of recent gradients (momentum) and gives every weight its own
  step size based on how large its gradients have been recently.
* Training uses mini-batches of 256 pairs, reshuffled each epoch.
* **Early stopping**: 5% of the training pairs are kept aside. After each epoch
  we measure the loss on them. If it has not improved for `patience` epochs we
  stop and restore the weights of the best epoch, so the network does not
  memorise the training set.

## Recommending

`recommend(user, n)` predicts a rating for every movie the user has **not rated
in the training set** and returns the `n` highest, as `(movie_id, rating)`.
Public methods use the 1-based ids from MovieLens; arrays are 0-based inside.

## Explaining a prediction

`explain(user, item)` uses **occlusion**: for each group of features we replace
the group's values by the typical training value ("neutral"), predict again, and
report `prediction_before - prediction_after`.

Groups are the reasons a person understands:

* each genre of the movie (the genre flag together with the user's taste for it)
* release era
* the user's average level (their mean and activity)
* movie popularity
* the movie's average rating

A positive number means that group pushed the rating up. Real output of
`python main.py recommend --user 196 --method nn` (fold 1, the first
recommendation, E.T. the Extra-Terrestrial), turned into a sentence by
`explain_nn`:

> Predicted 4.2 stars. Mainly because: you rate Fantasy above your average
> (+0.47), Children's raises the estimate here (+0.45), Sci-Fi raises the
> estimate here although you rate it below your average (+0.37).

How a genre reason is worded. The number in brackets is the genre group's
occlusion delta, which mixes the genre flag and the user's taste for it, so its
sign alone says nothing about the user's taste. The sentence therefore takes the
taste direction from the user's real taste value:

* taste and effect point the same way: "you rate Fantasy above your average (+0.47)";
* they point opposite ways (the genre flag outweighs the taste): "Sci-Fi raises
  the estimate here although you rate it below your average (+0.37)";
* the user has no taste for that genre (never rated it, taste exactly 0):
  "Children's raises the estimate here (+0.45)", with no claim about taste.

`scripts/check_explanations.py` checks every Top-10 explanation of all 943
users on fold 1 and counts stated tastes that contradict the real taste; it
prints 0. (The earlier wording took the direction from the delta; an audit of
the Top-10 lists of every 7th user on fold 1 found it wrong for 271 of 1,283
genre reasons, 21%.)

Limits: the numbers are not additive (the network is non-linear, so the deltas
need not sum to the total), and the user-activity and "average taste" features
are not shown as separate reasons: activity belongs to the user-level group,
average taste is left as background.

## How we evaluate (`src/evaluate.py`)

* Official splits `u1..u5` (each trains on 80,000 ratings, tests on 20,000).
  Every model is trained from scratch on each fold.
* **Candidates**: movies the user has not rated in the training set.
* **Relevant**: test movies rated 4 or more. Users with no relevant movie are
  skipped for the ranking metrics.
* **Precision@10**: the share of the 10 recommendations that are relevant.
  **Recall@10**: the share of the user's relevant movies that were found.
* **RMSE / MAE** on all test ratings, for models that predict ratings (the
  neural network, and the baselines). Naive Bayes does not predict ratings, so
  it gets "n/a".
* Results are the mean and standard deviation over the five folds, saved to
  `results/results.csv`.
* Baselines for context: Random, Popularity.
