# Human evaluation (blind plausibility study)

This folder holds a kit for the "performance by human judgement" part of the
project. No study has been run yet: there are **no human ratings** in this
repository until you collect them.

## What raters do

Raters are classmates, so they cannot judge whether a list suits *their own*
taste for a stranger. Instead they judge **plausibility**: for each of 10 real
MovieLens users they see the user's 5 highest- and 3 lowest-rated movies, then
three Top-10 lists (List A/B/C) and answer:

1. How well does this list fit this user? (1-5, for each list)
2. Which list fits best? (A/B/C)

In Part 2 they rate two one-sentence reasons per user (Reason X/Y):
"How clear and convincing is this reason?" (1-5).

The lists come from the neural network, Naive Bayes and Popularity, trained on
fold 1 with default settings. The order is shuffled differently for every user
and the sheet never names a method. Reasons come from the neural network and
Naive Bayes, each explaining its own top recommendation, also shuffled.

One limit to state in the report: the two reasons have recognisably different
wording ("Predicted 4.2 stars…" vs "Estimated chance of a like…"), so Part 2
is blind to the method's name, not to its style.

## Files

| File | Who sees it |
|---|---|
| `rater_sheet.md` | raters (print it, or export to PDF) |
| `ratings_template.csv` | the coordinator, who types in each rater's answers |
| `answer_key.csv` | coordinator only: maps List A/B/C and Reason X/Y to the methods |
| `tabulate.py` | coordinator: summarises the filled ratings |

Regenerate the sheet, template and key with `python main.py human-eval`
(same users and order every time for the same `--seed`).

## Running the study (5 to 10 raters, about 20 minutes each)

1. Print `rater_sheet.md` once per rater. Do not show `answer_key.csv` and do
   not say which method made which list.
2. Explain the task in one minute: "judge whether each list fits *this* user,
   using their profile, not your own taste". Raters work alone, without
   discussing.
3. Copy `ratings_template.csv` to `ratings.csv`. For each rater, append one
   copy of the template rows, put the rater's code in `rater`, and type in:
   * `fit_score` on every row (List A, B, C),
   * `best_list`, `clarity_X` and `clarity_Y` on the **List A row** of each
     user (leave them blank on the B and C rows).
4. Run `python human_eval/tabulate.py`. It prints the number of raters, the
   mean fit score per method, the "best list" wins per method and the mean
   explanation clarity per method. It refuses to run on the empty template.
5. Report the numbers exactly as printed, with the number of raters, and keep
   the filled `ratings.csv` in the repository as evidence.
