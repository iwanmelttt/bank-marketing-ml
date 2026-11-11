# Bank Marketing: End-to-End ML Consultancy Project

Predicting whether a bank client will subscribe to a term deposit **before** they are called, using the public [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) (`bank-additional-full.csv`, 41,188 rows, 20 features).

The project covers the full lifecycle: problem framing, a supervised vs unsupervised comparison, EDA and data quality, a leakage-safe preprocessing pipeline with controlled experiments, three tuned base models, voting and stacking ensembles, and an evaluation with bootstrap confidence intervals.

## Headline results (held-out test set, 8,236 clients)

| Model | ROC-AUC | PR-AUC | F1 | MCC | Subscribers reached by calling top 20% |
|---|---|---|---|---|---|
| Logistic Regression | 0.800 | 0.446 | 0.499 | 0.431 | 64.3% |
| Random Forest | 0.811 | 0.482 | 0.526 | 0.462 | 65.7% |
| Histogram Gradient Boosting | 0.813 | 0.486 | 0.525 | 0.460 | 66.3% |
| **Soft Voting (RF + HGB)** | **0.813** | **0.487** | **0.530** | **0.467** | 66.1% |
| Stacking (RF + HGB) | 0.813 | 0.486 | 0.529 | 0.465 | 66.0% |

F1 and MCC use thresholds chosen on out-of-fold training predictions. **Recommendation:** the soft-voting ensemble. It beats the cross-validated best single model by a small but statistically reliable margin (+0.005 PR-AUC, paired bootstrap 95% CI +0.001 to +0.009), and calling its top-scored 20% of clients reaches two-thirds of all subscribers.

## Repository structure

```
bank-marketing-ml/
├── data/
│   ├── raw/bank-additional-full.csv          # original UCI file (semicolon-separated)
│   └── processed/bank_marketing_cleaned.csv  # cleaned and feature-engineered dataset
├── notebooks/
│   └── bank_marketing_end_to_end.ipynb       # executed notebook, Parts A to H
├── src/
│   ├── config.py          # paths, seed, column groups
│   ├── data_loading.py    # load raw file, split X / y
│   ├── eda.py             # descriptive stats, quality checks, EDA figures
│   ├── preprocessing.py   # cleaning, feature engineering, leakage-safe pipeline
│   ├── experiments.py     # preprocessing experiments, leakage benchmark, k-means probe
│   ├── models.py          # base models, search spaces, voting and stacking
│   ├── evaluation.py      # metrics, thresholds, bootstrap CIs, plots
│   └── train.py           # end-to-end training script
├── reports/figures/       # all figures used in the report
├── results/               # CSV/JSON metrics (fitted models are git-ignored)
├── report/
│   ├── technical_report.md
│   └── technical_report.docx
├── requirements.txt
└── README.md
```

## How to run

Python 3.10 or later.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Option 1: run the whole pipeline as a script (about 6 minutes on 4 cores)
python -m src.train

# Option 2: open and re-run the notebook (about 8 minutes)
jupyter notebook notebooks/bank_marketing_end_to_end.ipynb
```

Both write metrics to `results/` and figures to `reports/figures/`. The random seed is fixed at 42, so results reproduce exactly with the same library versions (tested with scikit-learn 1.9.1, imbalanced-learn 0.14.2, pandas 3.0.5).

## Cleaned dataset

`data/processed/bank_marketing_cleaned.csv` holds 41,176 rows and 20 columns. Compared with the raw file:

| Change | Reason |
|---|---|
| 12 exact duplicate rows removed | avoid double weighting and train/test overlap |
| `duration` dropped | target leakage: call length is only known after the call |
| `pdays` replaced by `prev_contacted` (0/1) | 999 is a sentinel for "never contacted" (96% of rows) |
| `campaign` replaced by `campaign_log` = log(1 + campaign) | skewness 4.8 reduced to 1.4 |
| `default` = 'yes' (3 rows) merged into 'unknown' | level too rare to estimate |
| dots in column names replaced by underscores | e.g. `emp.var.rate` to `emp_var_rate` |

'unknown' values are deliberately kept as a category. Encoding, scaling and the correlation filter are **not** applied to the CSV; they are fitted inside the model pipeline on training data only, to avoid leakage.

## Method summary

* **Split:** stratified 80/20 train/test; all choices made with 5-fold stratified CV on the training set; test set used once.
* **Preprocessing (chosen by experiment, see Part D):** 'unknown' kept as a level; one-hot encoding with rare levels grouped; standard scaling; correlation filter (|ρ| > 0.9) for Logistic Regression only; class weights plus a tuned decision threshold for imbalance. PCA, SMOTE and random undersampling were tested and rejected because each lowered PR-AUC.
* **Base models:** Logistic Regression, Random Forest and Histogram Gradient Boosting, each tuned by randomised search optimising PR-AUC.
* **Ensembles:** soft voting and stacking (logistic meta-learner on out-of-fold probabilities) of the two best models by CV.
* **Evaluation:** ROC-AUC, PR-AUC, precision, recall, F1, MCC, balanced accuracy, Brier score, cumulative gains; 1,000-sample bootstrap CIs and paired bootstrap tests.

## Notes on the brief

The brief provided had Parts A to E and H. The ensemble work is presented as **Part F**. If the official brief has a **Part G** (for example hyperparameter tuning or interpretation), note that tuning is covered in Part E1 and model interpretation (permutation importance) in Part H4.

## Data source and licence

Moro, S., Cortez, P. and Rita, P. (2014) 'A data-driven approach to predict the success of bank telemarketing', *Decision Support Systems*, 62, pp. 22–31. Dataset available from the UCI Machine Learning Repository under a CC BY 4.0 licence (doi:10.24432/C5K306).
