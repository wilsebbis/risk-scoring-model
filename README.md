# Risk Scoring Model

A production-grade fraud detection pipeline that trains an XGBoost classifier on severely imbalanced transaction data, applies cost-aware threshold optimization under a hard **≤ 2 % false-positive-rate** constraint, and exports artifacts ready for batch inference and drift monitoring.

[![CI](https://github.com/wilsebbis/risk-scoring-model/actions/workflows/ci.yml/badge.svg)](https://github.com/wilsebbis/risk-scoring-model/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

---

## Key Capabilities

| Capability | Details |
|---|---|
| **Class-imbalance handling** | `scale_pos_weight`, AUCPR-driven early stopping |
| **Threshold optimization** | Grid search over 500 candidates; maximizes net savings while enforcing FPR ≤ 2 % |
| **Cost calibration** | Configurable `chargeback_cost` / `false_positive_cost` tradeoff |
| **3-way decision policy** | APPROVE / REVIEW / DECLINE with configurable review band |
| **Calibration** | Brier score + reliability-diagram data for probability sanity |
| **Confidence intervals** | Bootstrap 95 % CI on net savings |
| **Top-K capture** | % of fraud caught in top 1 % risk scores |
| **Drift monitoring** | PSI with explicit triage policy (OK → investigate → retrain) |
| **Warehouse training** | Optional Snowflake connector for query-based training sets |

---

## Decision Policy

The **model** produces a probability score. The **policy** converts scores into actions:

```
                        ┌─────────────┐
  Raw scores ──────────▶│  Threshold   │
  P(fraud | x)          │   Policy     │
                        └──────┬──────┘
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
          ┌─────────┐    ┌──────────┐    ┌──────────┐
          │ DECLINE  │    │  REVIEW  │    │ APPROVE  │
          │ score≥T₁ │    │ T₂≤s<T₁  │    │ score<T₂ │
          └─────────┘    └──────────┘    └──────────┘
```

- **T₁** (decline threshold) is selected by the optimizer under the FPR constraint.
- **T₂** (review threshold) is configurable — set `threshold.review_threshold` in config.
- The **model is not the policy.** The policy is constrained by business rules (FPR ceiling, cost ratio). Changing the policy does not require retraining.

---

## Repository Structure

```
risk-scoring-model/
├── configs/
│   └── default.yaml            # Training, threshold, cost, and artifact settings
├── scripts/
│   └── generate_synthetic_data.py
├── src/risk_scoring/
│   ├── config.py               # YAML config loader & validation
│   ├── data.py                 # Data ingestion (CSV / Snowflake)
│   ├── evaluation.py           # Threshold search, cost-curve, calibration, bootstrap CI
│   ├── modeling.py             # XGBoost wrapper
│   ├── monitoring.py           # PSI drift + triage policy + live performance
│   ├── score.py                # Batch scoring CLI with 3-way decisions
│   └── train.py                # End-to-end training entrypoint
├── tests/
│   ├── conftest.py
│   ├── test_evaluation.py
│   └── test_monitoring.py
├── artifacts/                  # Model, metrics, threshold (gitignored)
├── data/                       # Raw CSVs (gitignored)
├── EVALUATION.md               # Dataset, split methodology, metric definitions
├── MODEL_CARD.md               # Intended use, limitations, fairness, rollback
├── .github/workflows/ci.yml
├── pyproject.toml
├── requirements.txt
└── requirements-test.txt
```

---

## Getting Started

### Prerequisites

- Python ≥ 3.10
- (Optional) A Snowflake account for warehouse-based training

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Generate Synthetic Data

```bash
python scripts/generate_synthetic_data.py --rows 250000
```

Creates `data/transactions.csv` with realistic class imbalance.

### Train

```bash
python -m risk_scoring.train --config configs/default.yaml
```

Outputs to `artifacts/`:

| Artifact | Description |
|---|---|
| `risk_model.joblib` | Serialized XGBoost model + threshold metadata |
| `metrics.json` | All evaluation metrics including CI, calibration, top-K |
| `threshold.json` | Optimal threshold and operating-point metadata |
| `threshold_curve.csv` | Full precision / recall / net-savings curve |

### Score

```bash
python -m risk_scoring.score \
  --model artifacts/risk_model.joblib \
  --input data/transactions.csv \
  --output artifacts/scored_transactions.csv \
  --threshold 0.87
```

### Score — 3-Way Decision Mode

```bash
python -m risk_scoring.score \
  --model artifacts/risk_model.joblib \
  --input data/transactions.csv \
  --output artifacts/scored_transactions.csv \
  --threshold 0.87 \
  --review-threshold 0.50
```

Emits a `decision` column: `DECLINE` (≥ 0.87), `REVIEW` (0.50–0.87), `APPROVE` (< 0.50).

### Monitor

```bash
python -m risk_scoring.monitoring \
  --baseline artifacts/scored_transactions.csv \
  --current data/new_transactions.csv
```

Produces a PSI drift report with explicit triage actions:

| PSI | Action |
|---|---|
| < 0.10 | **OK** — no action needed |
| 0.10 – 0.25 | **Investigate** — review feature distributions |
| > 0.25 | **Retrain** or rollback to previous model |

> **Label delay:** Fraud labels typically arrive 30–90 days post-transaction. Recall metrics within this window are provisional. PSI (label-free) provides the earliest drift signal.

---

## Configuration

All settings live in `configs/default.yaml`. Key sections:

```yaml
training:
  split_mode: stratified   # or "time" for temporal evaluation

threshold:
  max_false_positive_rate: 0.02   # Hard constraint
  candidate_steps: 500
  review_threshold: null          # Set e.g. 0.5 for 3-way decisions

costs:
  chargeback_cost: 150.0          # $ lost per missed fraud
  false_positive_cost: 25.0       # $ lost per false decline
```

See [`configs/default.yaml`](configs/default.yaml) for the full schema.

---

## Snowflake Integration

Local dev uses CSV. The Snowflake path exists for production realism.

Set `data.source: snowflake` and `data.snowflake_query` in the config:

```yaml
data:
  source: snowflake
  snowflake_query: |
    SELECT t.*, f.is_fraud
    FROM analytics.transactions t
    LEFT JOIN labels.fraud_flags f USING (transaction_id)
    WHERE t.created_at >= DATEADD(month, -6, CURRENT_DATE())
```

Export credentials:

```bash
export SNOWFLAKE_ACCOUNT=...
export SNOWFLAKE_USER=...
export SNOWFLAKE_PASSWORD=...
export SNOWFLAKE_WAREHOUSE=...
export SNOWFLAKE_DATABASE=...
export SNOWFLAKE_SCHEMA=...
```

The pipeline reads directly from your warehouse — no CSV export required.

---

## Testing

```bash
pip install -r requirements-test.txt
pytest -q
```

CI runs automatically on every push and pull request via GitHub Actions.

---

## Documentation

| Document | Purpose |
|---|---|
| [EVALUATION.md](EVALUATION.md) | Dataset description, split methodology, metric definitions, headline numbers |
| [MODEL_CARD.md](MODEL_CARD.md) | Intended use, limitations, fairness, leakage risks, rollback plan |

---

## License

This project is provided as-is for educational and portfolio purposes.