# Evaluation Report

## Dataset

**Source:** deterministic synthetic transaction generator (`scripts/generate_synthetic_data.py`).

Features are drawn from parametric distributions calibrated to real-world card transaction patterns:

| Feature | Distribution | Fraud Signal |
|---|---|---|
| `amount` | LogNormal(μ=3.4, σ=0.9) | Higher amounts ↑ risk |
| `device_age_days` | Gamma(k=2.1, θ=90) | New device < 10d ↑ risk |
| `velocity_1h` | Poisson(λ=0.8) + burst injection | High velocity ↑ risk |
| `country_risk` | Beta(1.4, 7.0) | Higher ↑ risk |
| `email_risk` | Beta(1.2, 8.5) | Higher ↑ risk |
| `ip_proxy` | Bernoulli(0.05) | Proxy ↑ risk |
| `card_present` | Bernoulli(0.66) | Present ↓ risk |
| `recent_declines` | Poisson(0.15), clipped | More declines ↑ risk |
| `weekend` | Bernoulli(0.28) | Weekend slightly ↑ risk |
| `merchant_category` | Categorical (5 classes) | Contextual signal |
| `channel` | Categorical (3 classes) | Contextual signal |

Fraud labels are generated via a sigmoid logit model with interaction terms, producing realistic class imbalance (~3–5 % positive rate).

> [!NOTE]
> Synthetic data intentionally lacks temporal correlation and concept drift. For production use, replace with real transaction data and consider `split_mode: time`.

---

## Split Methodology

| Mode | Method | When to Use |
|---|---|---|
| `stratified` (default) | `sklearn.train_test_split` with `stratify=y` | Standard offline evaluation; preserves class balance across splits |
| `time` | Chronological split — last `test_size` fraction of rows | Temporal evaluation; demonstrates leakage awareness |

Both modes use a 3-way split: **train → validation → test**.

- **Validation set:** used for XGBoost early stopping only; never touches threshold selection.
- **Test set:** used for all reported metrics, threshold sweep, and confidence intervals.

---

## Metric Definitions

### Chargeback Loss Baseline

The cost of doing nothing (approving all transactions):

```
baseline_cost = count(fraud in test) × chargeback_cost
```

### Net Savings

Savings from deploying the model at the selected threshold:

```
model_cost    = (FN × chargeback_cost) + (FP × false_positive_cost)
net_savings   = baseline_cost − model_cost
net_savings_% = net_savings / baseline_cost
```

### Decline Rate

Fraction of all transactions flagged (TP + FP) out of total:

```
decline_rate = (TP + FP) / N
```

Fraud teams ask: "What did this model do to our approval rate?" Decline rate answers that directly.

### FPR Constraint

The threshold is selected to maximize net savings **subject to FPR ≤ 2 %**. This hard constraint protects customer experience.

### Top-1 % Capture

Fraction of total fraud in the top 1 % highest-risk scores. Useful as a ranking-quality sanity check independent of any threshold choice.

### Bootstrap 95 % CI

1,000 bootstrap resamples of the test set. Reports:

```
net_savings_pct: X% ± Y% (95% CI)
```

### Calibration (Brier Score)

Brier score measures how well raw model scores approximate true probabilities. A reliability diagram (bin-level `mean_predicted` vs `fraction_positive`) is available in `metrics.json → calibration.bins`.

---

## Headline Numbers

> Run `python -m risk_scoring.train --config configs/default.yaml` to populate.

After training, the key numbers to cite are:

| Metric | Value |
|---|---|
| AUPRC | See `metrics.json → ranking_metrics.auprc` |
| FPR at selected threshold | See `metrics.json → selected_threshold.selected_metrics.false_positive_rate` |
| Recall at selected threshold | See `metrics.json → selected_threshold.selected_metrics.recall` |
| Net savings % of baseline | See `metrics.json → selected_threshold.selected_metrics.net_savings_pct_of_baseline` |
| 95 % CI | See `metrics.json → bootstrap_ci_95` |
| Decline rate | See `metrics.json → selected_threshold.selected_metrics.decline_rate` |
| Top-1 % capture | See `metrics.json → top_k_capture_1pct` |
| Brier score | See `metrics.json → calibration.brier_score` |

### Resume Bullet Template

> "Shipped XGBoost fraud scoring pipeline with threshold policy enforcing < 2 % FPR, reducing chargeback losses by **X % ± Y %** (95 % CI) at a **Z %** decline rate."
