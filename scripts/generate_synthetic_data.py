from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic card transaction data")
    parser.add_argument("--rows", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/transactions.csv")
    return parser.parse_args()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    n = args.rows

    amount = rng.lognormal(mean=3.4, sigma=0.9, size=n)
    device_age_days = np.clip(rng.gamma(shape=2.1, scale=90.0, size=n), 0, 2500)
    velocity_1h = rng.poisson(lam=0.8, size=n) + (rng.random(n) < 0.03) * rng.integers(3, 12, size=n)
    country_risk = rng.beta(a=1.4, b=7.0, size=n)
    email_risk = rng.beta(a=1.2, b=8.5, size=n)
    ip_proxy = rng.binomial(1, 0.05, size=n)
    card_present = rng.binomial(1, 0.66, size=n)
    recent_declines = np.clip(rng.poisson(0.15, size=n), 0, 6)
    weekend = rng.binomial(1, 0.28, size=n)

    amount_ln = np.log1p(amount)
    new_device = (device_age_days < 10).astype(int)

    fraud_logit = (
        -4.7
        + 0.65 * amount_ln
        + 0.25 * velocity_1h
        + 2.1 * country_risk
        + 1.9 * email_risk
        + 1.1 * ip_proxy
        + 0.55 * new_device
        + 0.25 * recent_declines
        - 0.35 * card_present
        + 0.2 * weekend
        + 0.7 * ((velocity_1h >= 5) & (ip_proxy == 1))
    )

    fraud_probability = np.clip(_sigmoid(fraud_logit), 0, 0.98)
    is_fraud = rng.binomial(1, fraud_probability)

    df = pd.DataFrame(
        {
            "transaction_id": np.arange(1, n + 1),
            "amount": amount.round(2),
            "device_age_days": device_age_days.round(1),
            "velocity_1h": velocity_1h,
            "country_risk": country_risk.round(4),
            "email_risk": email_risk.round(4),
            "ip_proxy": ip_proxy,
            "card_present": card_present,
            "recent_declines": recent_declines,
            "weekend": weekend,
            "merchant_category": rng.choice(
                ["retail", "travel", "digital_goods", "food", "gaming"],
                size=n,
                p=[0.34, 0.14, 0.18, 0.24, 0.10],
            ),
            "channel": rng.choice(
                ["web", "mobile", "pos"],
                size=n,
                p=[0.48, 0.34, 0.18],
            ),
            "is_fraud": is_fraud,
        }
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Wrote {len(df):,} rows to {output_path}")
    print(f"Fraud rate: {df['is_fraud'].mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
