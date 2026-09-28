# Airline Review Recommendation Analytics

This project analyses which parts of an airline experience are most closely associated with a customer recommendation. It compares structured service ratings with review text and tests whether the models generalise to airlines excluded from training.

The original academic analysis covered **23,171 reviews across 497 airlines** and compared structured service ratings, TF-IDF text and combined models. For this repository, I corrected the most important technical weakness in that submission: airline-level imputation is now fitted **only on the training data**, with a training-derived global fallback for unseen airlines.

## Analysis covered

- customer recommendation classification;
- group-aware and random holdout evaluation;
- training-only, airline-level median imputation;
- TF-IDF text modelling alongside service ratings;
- comparable accuracy, precision, recall, F1 and ROC-AUC reporting;
- feature and coefficient evidence translated into commercial action;
- separate reporting for the original academic findings and the synthetic software check.

![Synthetic validation model comparison](figures/model_comparison.png)

> **Data note:** the chart above checks the pipeline on synthetic data; it is not a claim about real airline performance. The original university workbook is not redistributed and was not available when this repository was prepared. The code can be rerun when an authorised copy of the source file is restored.

## Leakage correction

The submitted workflow calculated airline medians before splitting the dataset. That allowed test-set information to influence training-time imputation and compromised both the random and airline-holdout evaluations.

The public pipeline instead follows this sequence:

```text
split reviews
    |
    v
fit airline medians on training rows only
    |
    +--> known airline: training airline median
    +--> unseen airline: training global median
    |
    v
fit model on transformed training set
    |
    v
evaluate untouched test set
```

The included test deliberately creates an unseen airline and confirms that its missing ratings use the training global fallback.

## Modelling tracks

| Track | Inputs | Purpose |
|---|---|---|
| Numerical logistic regression | Eight service ratings | Transparent driver direction |
| Numerical random forest | Eight service ratings | Non-linear benchmark and importance |
| Text logistic regression | TF-IDF review text | Language and complaint signal |
| Combined logistic regression | Ratings + TF-IDF | Test whether text adds predictive lift |
| Airline holdout | Unseen airlines | Generalisation beyond known carriers |

![Synthetic validation feature importance](figures/numeric_feature_importance.png)

## Business interpretation

The original analysis consistently identified perceived value, ground service and cabin service as commercially important. The corrected evaluation makes those findings easier to test. The output can be used to:

- prioritise service attributes associated with recommendation;
- compare the incremental value of free text against structured ratings;
- audit errors rather than treating one accuracy number as proof;
- distinguish diagnosis after a review from a pre-travel loyalty prediction.

## Repository guide

| Path | Purpose |
|---|---|
| `src/airline_analytics.py` | Cleaning, training-only imputation, modelling and metrics |
| `scripts/generate_demo_data.py` | Reproducible synthetic validation data |
| `scripts/run_analysis.py` | Command-line workflow for CSV or Excel input |
| `tests/` | Leakage and output checks |
| `data/demo_airline_reviews.csv` | Synthetic data only |
| `outputs/` | Clearly labelled demo metrics and evidence |

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/generate_demo_data.py
python scripts/run_analysis.py --data data/demo_airline_reviews.csv --data-label synthetic_demo
python -m unittest discover -s tests -v
```

The runner accepts `.csv` and `.xlsx`. To rebuild the real case, place the authorised university dataset outside version control and pass its path with `--data`.

## Limitations

- Recommendation is observed after the experience; this is diagnosis, not a pre-flight causal prediction.
- Review platforms are self-selected and may not represent all passengers.
- Structured ratings and recommendation can reflect the same underlying judgement.
- Synthetic validation proves the software path, not the historic performance claims.

## Author

**Muhammad Ahmed Shoaib**<br>
Customer analytics, machine learning and commercial decision support.
