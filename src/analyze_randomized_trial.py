"""Analyze a future member-level randomized retention test.

Expected CSV columns: loyalty_number, arm, flight_within_90d.
`arm` must be `control` or `treatment`; outcome is 0/1. Each member appears once.
This script cannot estimate a treatment effect from the supplied historical
data because that data contains no randomized campaign assignment.

Run after a real test:
    python src/analyze_randomized_trial.py path/to/randomized_results.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import norm


def wilson_interval(successes: int, total: int, z: float) -> tuple[float, float]:
    if total == 0:
        raise ValueError("Each arm must contain at least one assigned member.")
    rate = successes / total
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    radius = z * ((rate * (1 - rate) / total + z**2 / (4 * total**2)) ** 0.5) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def analyze(frame: pd.DataFrame) -> dict:
    required = {"loyalty_number", "arm", "flight_within_90d"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame["loyalty_number"].duplicated().any():
        raise ValueError("Expected one row per randomized loyalty_number (member-level assignment).")
    if set(frame["arm"].dropna().unique()) - {"control", "treatment"}:
        raise ValueError("arm must contain only 'control' and 'treatment'.")
    if set(frame["flight_within_90d"].dropna().unique()) - {0, 1}:
        raise ValueError("flight_within_90d must be coded 0/1.")
    if frame[["arm", "flight_within_90d"]].isna().any().any():
        raise ValueError("Resolve missing arm or outcome values before analysis; do not silently drop them.")

    control = frame.loc[frame["arm"].eq("control"), "flight_within_90d"].astype(int)
    treatment = frame.loc[frame["arm"].eq("treatment"), "flight_within_90d"].astype(int)
    if control.empty or treatment.empty:
        raise ValueError("Both randomized arms must be represented.")

    n0, n1 = len(control), len(treatment)
    x0, x1 = int(control.sum()), int(treatment.sum())
    p0, p1 = x0 / n0, x1 / n1
    difference = p1 - p0
    pooled = (x0 + x1) / (n0 + n1)
    se_null = (pooled * (1 - pooled) * (1 / n0 + 1 / n1)) ** 0.5
    z_stat = difference / se_null if se_null else 0.0
    p_value = float(2 * norm.sf(abs(z_stat)))

    z_95 = float(norm.ppf(0.975))
    lo0, hi0 = wilson_interval(x0, n0, z_95)
    lo1, hi1 = wilson_interval(x1, n1, z_95)
    ci_low = difference - ((p1 - lo1) ** 2 + (hi0 - p0) ** 2) ** 0.5
    ci_high = difference + ((hi1 - p1) ** 2 + (p0 - lo0) ** 2) ** 0.5

    return {
        "analysis": "intent-to-treat difference in 90-day flight return rate",
        "control_n": n0,
        "control_return_rate": p0,
        "treatment_n": n1,
        "treatment_return_rate": p1,
        "absolute_lift": difference,
        "newcombe_95pct_ci_low": max(-1.0, ci_low),
        "newcombe_95pct_ci_high": min(1.0, ci_high),
        "two_sided_p_value": p_value,
        "interpretation_note": "Report this as causal only if assignment was randomized as specified and outcome capture is complete.",
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python src/analyze_randomized_trial.py <assignment_and_outcome.csv>")
    path = Path(sys.argv[1])
    result = analyze(pd.read_csv(path))
    print(pd.Series(result).to_string())


if __name__ == "__main__":
    main()
