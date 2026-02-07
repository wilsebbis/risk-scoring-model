# Fraud / Risk Scoring Model

Production-style fraud scoring pipeline built for the exact interview deep-dives that matter in fintech and trust & safety:

- XGBoost classifier for severe class imbalance
- Explicit threshold optimization with a hard `false_positive_rate <= 2%` constraint
- Cost-aware optimization with chargeback vs false-decline tradeoff
- Monitoring report with score drift (PSI) and optional live labeled performance
- Snowflake ingestion path for warehouse-native training

## Why this project is elite

- Directly tied to a monetizable KPI (chargeback loss reduction)
- Precision-recall and cost-curve decisions are first-class, not afterthoughts
- Clear path to production ownership: train, tune threshold, monitor drift
- Great talking points for ML + security + fintech hiring loops

## Repository Layout

- `configs/default.yaml`: training, threshold, costs, artifacts config
- `scripts/generate_synthetic_data.py`: deterministic synthetic transaction generator
- `src/risk_scoring/train.py`: end-to-end training and artifact export
- `src/risk_scoring/score.py`: batch scoring CLI for inference datasets
- `src/risk_scoring/monitoring.py`: PSI drift + live performance report

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/generate_synthetic_data.py --rows 250000
python -m risk_scoring.train --config configs/default.yaml
python -m risk_scoring.score --model artifacts/risk_model.joblib --input data/transactions.csv --output artifacts/scored_transactions.csv --threshold 0.87
```

Expected artifacts in `artifacts/`:

- `risk_model.joblib`
- `metrics.json`
- `threshold.json`
- `threshold_curve.csv`

## Snowflake Training

Set `data.source: snowflake` and provide `data.snowflake_query` in config.

Required environment variables:

- `SNOWFLAKE_ACCOUNT`
- `SNOWFLAKE_USER`
- `SNOWFLAKE_PASSWORD`
- `SNOWFLAKE_WAREHOUSE`
- `SNOWFLAKE_DATABASE`
- `SNOWFLAKE_SCHEMA`
- Optional: `SNOWFLAKE_ROLE`

## Run Tests

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-test.txt
pytest -q
```

## Interview-Ready Deep Dives

1. Threshold policy:
   - Why optimize by net savings instead of pure AUC
   - How FPR ceiling (`<2%`) protects customer experience
2. Cost calibration:
   - How `chargeback_cost` and `false_positive_cost` shape decisions
3. Monitoring:
   - PSI thresholds for drift triage
   - Label delay implications for fraud recall tracking
4. Security posture:
   - Feature hygiene and leakage controls
   - Human-in-the-loop review for high-risk edge bands

## Suggested KPI framing

Use this template for portfolio or resume bullets after running on real data:

- "Shipped XGBoost fraud scoring pipeline in Python + Snowflake with threshold policy enforcing `<2%` false positives, reducing chargeback losses by `X%` at stable approval rate."
