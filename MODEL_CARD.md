# Model Card — Fraud Risk Scoring Model

## Overview

| Field | Value |
|---|---|
| **Model type** | XGBoost binary classifier (`binary:logistic`) |
| **Task** | Transaction-level fraud detection |
| **Output** | Probability score ∈ [0, 1] |
| **Decision policy** | Score → threshold → APPROVE / REVIEW / DECLINE |

---

## Intended Use

- **Primary:** offline batch scoring of card transactions for chargeback risk triage.
- **Deployment:** behind a decision policy with a hard FPR ≤ 2 % constraint and an optional manual-review band.
- **Users:** fraud operations teams, risk engineers.

---

## Out of Scope / Non-Goals

- Real-time sub-10 ms inference (use a feature store + lightweight model for that).
- Account-level risk (this model is transaction-scoped).
- Regulated credit decisioning (this is a fraud signal, not a lending decision).

---

## Training Data

- Default: synthetic transactions from `scripts/generate_synthetic_data.py` (250 K rows, ~3–5 % fraud rate).
- Production: replace with your warehouse query via `data.source: snowflake` in config.
- See [EVALUATION.md](EVALUATION.md) for feature definitions and data generation methodology.

---

## Limitations

1. **Synthetic data bias.** The default model is trained on parametric synthetic data. Feature relationships and imbalance ratios may not transfer to production distributions.
2. **No temporal modeling.** Features are point-in-time; the model does not capture sequential patterns (e.g., velocity streaks across sessions).
3. **Static threshold.** The threshold is calibrated once at training time. Concept drift will degrade the operating point.
4. **Single-model architecture.** No ensembling, stacking, or multi-stage cascade. Suitable for a baseline; production systems may layer additional models.

---

## Fairness Considerations

- No protected-class features (age, gender, race, ZIP) are included in the default feature set.
- **Proxy risk:** `country_risk` and `merchant_category` could correlate with protected demographics. Production deployments should audit disparate impact on flagging rates across demographic groups.
- Recommendation: add a fairness audit step before production rollout, reporting flag rates by demographic segment.

---

## Leakage Risks

| Risk | Mitigation |
|---|---|
| Future features in training | Use `split_mode: time` to simulate production ordering |
| Label leakage | Target column (`is_fraud`) is explicitly excluded from features |
| Score-threshold circularity | Threshold is selected on test set after model training; test set never touches training |

---

## Monitoring & Rollback

### Drift Detection

Population Stability Index (PSI) on score distribution:

| PSI | Action |
|---|---|
| < 0.10 | OK — no action |
| 0.10 – 0.25 | Investigate — review feature distributions |
| > 0.25 | Retrain or rollback to previous model |

### Label Delay

Fraud labels typically arrive 30–90 days post-transaction. Recall metrics computed within this window are **provisional**. Monitor PSI (label-free) for early drift signals.

### Rollback Plan

1. Model artifacts are versioned by `trained_at_utc` in the joblib bundle.
2. Rollback: point the scoring pipeline to the previous `risk_model.joblib`.
3. No database migrations required — the model is stateless.

---

## Ethical Considerations

- False positives (declined legitimate transactions) cause customer friction and potential lost revenue.
- False negatives (missed fraud) cause direct financial loss.
- The cost ratio (`chargeback_cost` / `false_positive_cost`) makes this tradeoff explicit and auditable.
- The FPR constraint exists specifically to bound customer harm.
