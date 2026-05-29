# DataCrunch Equity Market Neutral

My submission for the [CrunchDAO DataCrunch Equity Market Neutral](https://www.crunchdao.com) competition.
Goal: predict a per-stock cross-sectional signal on the Russell 3000 universe, scored by **per-moon Pearson correlation**.

Everything lives in [`submission.ipynb`](submission.ipynb): EDA → metric study → modeling → local evaluation.


## How to run

```bash
# 1. Create a virtualenv and install deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Set up the Crunch CLI (replace <YOUR_TOKEN> with yours)
#    Token available at https://www.crunchdao.com (Account > API tokens)
crunch setup-notebook datacrunch-2 <YOUR_TOKEN>

# 3. Open the notebook and run all cells
jupyter notebook submission.ipynb
```

The first run will download the reduced dataset (~a few hundred MB) into `data/`, train the model,
and write it to `resources/model.joblib`.

## Notebook structure

- **Phase A — EDA.** Target distribution, septile-encoded features, two notable clusters ("stars" = ~15 monotone features with strong signal, "gems" = ~15 non-monotone "reversal at extremes" features), temporal stability, linear (IC sharpe) vs non-linear (MI) signal, hierarchical clustering (~30% of features deduplicable at corr ≥ 0.95).
- **Phase B — Metric study.** Monte-Carlo simulator of the per-moon Pearson on a ternary target — sign accuracy on movers dominates.
- **Phase C — Modeling.** LightGBM regressor, hyperparameters tuned via Optuna (hardcoded in `train()`). "Extreme reversal" binary features explored but not retained in the final model.
- **Phase D — Local evaluation.** Per-moon Pearson, per-regime breakdown, feature importance.

## What was removed from this clean notebook

To keep the repo focused, the following development-time sections were dropped from `submission.ipynb`:

- **B.2** — Walk-forward validation harness (`walk_forward_score`, baseline LGBM run).
- **C.1.1** — Family aggregation features (negative result: no measurable gain).
- **C.2.1** — Optuna hyperparameter search (the best params it found are hardcoded in the final `train()`).


