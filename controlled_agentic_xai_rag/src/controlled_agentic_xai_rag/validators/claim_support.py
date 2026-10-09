"""Claim support aggregation."""

from __future__ import annotations

import pandas as pd


def claim_support_summary(claims_df: pd.DataFrame) -> dict:
    if claims_df.empty:
        return {"check": "claim_support", "passed": True, "fully_supported_rate": 1.0, "unsupported_count": 0}
    total = len(claims_df)
    fully = int((claims_df["support_status"] == "fully_supported").sum())
    unsupported = int(claims_df["support_status"].isin(["unsupported", "contradicted"]).sum())
    return {
        "check": "claim_support",
        "passed": unsupported == 0,
        "fully_supported_rate": round(fully / total, 4) if total else 1.0,
        "unsupported_count": unsupported,
        "total_claims": total,
    }
