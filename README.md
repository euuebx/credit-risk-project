# Credit Risk Scoring Engine

A C++17 CLI with two modes:

1. **Heuristic scorer** — scores customers from a CSV with a hand-weighted formula
   (debt-to-income 40%, credit utilization 35%, late-payment rate 25%) passed through a
   saturating exponential transform.
2. **Validated model** — fits a logistic regression (batch gradient descent, implemented from
   scratch) on labeled data, evaluates it on a held-out test set, and compares it against a
   hand-picked-weight baseline.

## Why this exists

The hand-picked 40/35/25 weights are a *baseline*, not a result. The `--train` mode answers
the obvious follow-up — "why those weights?" — with data: the fitted model learns its
coefficients from labeled examples, and its ROC AUC is reported next to the baseline's AUC on
the same held-out split.

## Data

The project ships only real data:

- `data/german_credit.csv` — UCI Statlog German Credit Data (1000 rows, 7 numeric
  features, binary label: 1 = high risk / defaulted, 0 = low risk).
  Source: <https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/>
- `data/lending_club_customers.csv` — 50,000 real Lending Club loans in the
  customer format, for the heuristic scorer.
- `data/lending_club_labeled.csv` — the same loans as fourteen real features
  (the heuristic's three ratios plus income, loan amount, interest rate,
  installment, term, credit-line counts, public records, inquiries,
  revolving balance and credit-history age) and the real outcome label
  (1 = Charged Off, 0 = Fully Paid), for `--train`.

Both Lending Club files are derived from published
[LoanStats](https://www.lendingclub.com/info/download-data.action) data:

| Feature | Lending Club field(s) |
|---|---|
| debt-to-income | `dti` (debt-to-income ratio) |
| credit utilization | `revol_util` (revolving utilization) |
| late-payment rate | `delinq_2yrs` (delinquencies, past 2 years) ÷ 12 |
| annual income / total debt | `annual_inc`; total debt = `dti` × `annual_inc` |
| credit limit / credit used | `total_rev_hi_lim`; `revol_bal` |
| loan amount, interest rate, installment, term | `loan_amnt`, `int_rate`, `installment`, `term` |
| credit lines, records, inquiries | `open_acc`, `total_acc`, `pub_rec`, `inq_last_6mths` |
| credit-history age | `earliest_cr_line` |

Loans with any other status (Current, Late, In Grace Period, ...) are excluded,
so the label is a real, observable outcome. On this data `--train` compares the
fitted model against the 40/35/25 heuristic applied to its three ratios: the
heuristic only ever sees those three features, the fitted model sees all
fourteen.

## Build

Requires a C++17 compiler (g++). On Windows, use WSL or MinGW.

```bash
make
```

## Run

Heuristic mode — scores real Lending Club loans (use `--no-reports --quiet`
for large inputs, otherwise it writes one report file per loan):

```bash
./bin/credit_risk --input data/lending_club_customers.csv --out reports --no-reports --quiet
./bin/credit_risk --input data/lending_club_customers.csv --out reports --summary summary.json --no-reports
```

Train/evaluate mode on the real Lending Club outcomes (validates the 40/35/25
heuristic against real defaults):

```bash
./bin/credit_risk --train data/lending_club_labeled.csv
```

Train/evaluate mode:

```bash
./bin/credit_risk --train data/german_credit.csv
./bin/credit_risk --train data/german_credit.csv --test-size 0.2 --seed 42 --summary model.json
```

## Test

```bash
make test
```

## Output (train mode)

- Reproducible train/test split (default 80/20, seeded)
- AUC, accuracy and confusion matrix for the fitted model
- Learned coefficients (effect per standard deviation of each feature)
- Baseline AUC (the 40/35/25 heuristic on Lending Club data; hand-picked
  weights on min-max normalised features on other datasets)

## Layout

```
src/main.cpp                 CLI and both run modes
src/CSVReader.*              Customer CSV parsing (quoted fields, whitespace rows)
src/Scorer.*                 Hand-weighted heuristic
src/GermanData.*             Labeled dataset loader
src/LogisticRegression.*     Gradient-descent logistic regression
src/Metrics.*                ROC AUC (Mann-Whitney U), accuracy, confusion matrix
src/tests.cpp                Unit tests (make test)
```

## Limitations

- The German-credit model uses numeric features only; categorical fields (credit history,
  purpose, housing, ...) are future work and would need one-hot encoding.
- The heuristic treats missing income/limit as maximum risk and missing numeric fields as 0.
- No regularization in the optimizer; harmless with 7 features.
