"""Shared evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mean_std(series: pd.Series) -> tuple[float, float]:
    return float(series.mean()), float(series.std())


def summarize_by_group(df: pd.DataFrame, group_col: str, metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_col):
        row = {group_col: key}
        for c in metric_cols:
            if c in g.columns:
                m, s = mean_std(g[c].dropna())
                row[f"{c}_mean"] = round(m, 4)
                row[f"{c}_std"] = round(s, 4)
        rows.append(row)
    return pd.DataFrame(rows)
