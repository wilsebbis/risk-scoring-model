from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DataConfig


@dataclass
class PreparedData:
    X: pd.DataFrame
    y: pd.Series
    feature_columns: list[str]


def load_raw_dataset(cfg: DataConfig) -> pd.DataFrame:
    source = cfg.source.lower().strip()

    if source == "csv":
        df = pd.read_csv(cfg.csv_path)
        if df.empty:
            raise ValueError(f"CSV file is empty: {cfg.csv_path}")
        return df

    if source == "snowflake":
        if not cfg.snowflake_query:
            raise ValueError("'data.snowflake_query' is required when source=snowflake")
        return _load_from_snowflake(cfg.snowflake_query)

    raise ValueError(f"Unsupported data source: {cfg.source}")


def _load_from_snowflake(query: str) -> pd.DataFrame:
    try:
        import snowflake.connector
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "snowflake-connector-python is required for source=snowflake"
        ) from exc

    required_env = [
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_WAREHOUSE",
        "SNOWFLAKE_DATABASE",
        "SNOWFLAKE_SCHEMA",
    ]
    missing = [name for name in required_env if not os.getenv(name)]
    if missing:
        raise EnvironmentError(
            "Missing Snowflake environment variables: " + ", ".join(sorted(missing))
        )

    conn = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
        role=os.getenv("SNOWFLAKE_ROLE"),
    )

    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()

    if df.empty:
        raise ValueError("Snowflake query returned no rows.")
    return df


def _encode_feature_frame(df: pd.DataFrame, drop_cols: list[str]) -> pd.DataFrame:
    X = df.drop(columns=[col for col in drop_cols if col in df.columns]).copy()

    bool_cols = X.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=list(cat_cols), dummy_na=True)

    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].isna().any():
            median = X[col].median() if pd.api.types.is_numeric_dtype(X[col]) else 0.0
            X[col] = X[col].fillna(median)

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype(np.float32)

    return X


def prepare_features(df: pd.DataFrame, target_col: str, id_col: str | None = None) -> PreparedData:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset")

    y = df[target_col].astype(int)
    if y.nunique() != 2:
        raise ValueError(f"Target must be binary, but '{target_col}' has {y.nunique()} classes")

    drop_cols = [target_col]
    if id_col:
        drop_cols.append(id_col)

    X = _encode_feature_frame(df, drop_cols)

    return PreparedData(X=X, y=y, feature_columns=list(X.columns))


def prepare_inference_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_col: str | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    drop_cols: list[str] = []
    if target_col:
        drop_cols.append(target_col)
    if id_col:
        drop_cols.append(id_col)

    X = _encode_feature_frame(df, drop_cols)
    X = X.reindex(columns=feature_columns, fill_value=0.0)
    return X
