"""Leakage-aware 2017-to-2018 churn modeling and retention-test planning.

The source submission trained on full 2017-2018 activity and defined silent
churn from that same activity window. This script instead fixes the feature
cutoff at 2017-12-31 and predicts 2018 churn outcomes.

Run from the project root:
    python src/temporal_retention_analysis.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Datasets"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"
RANDOM_STATE = 42
AS_OF_YEAR = 2017
AS_OF_MONTH = 12


def _months_since(year: int, month: int, as_of_year: int, as_of_month: int) -> int:
    return (as_of_year - year) * 12 + as_of_month - month


def _longest_zero_run(values: np.ndarray) -> int:
    longest = current = 0
    for has_activity in values:
        if has_activity:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def build_cohort_and_features() -> tuple[pd.DataFrame, dict]:
    members = pd.read_csv(DATA_DIR / "Customer Loyalty History.csv")
    activity_raw = pd.read_csv(DATA_DIR / "Customer Flight Activity.csv")

    key = ["Loyalty Number", "Year", "Month"]
    duplicate_mask = activity_raw.duplicated(key, keep=False)
    duplicate_rows = int(activity_raw.duplicated(key).sum())
    duplicate_groups = int(activity_raw.loc[duplicate_mask, key].drop_duplicates().shape[0])
    duplicated = activity_raw.loc[duplicate_mask]
    conflicting_duplicate_groups = int(
        duplicated.groupby(key, dropna=False)[
            ["Total Flights", "Distance", "Points Accumulated", "Points Redeemed", "Dollar Cost Points Redeemed"]
        ].nunique(dropna=False).max(axis=1).gt(1).sum()
    )

    # The submission's documented policy keeps the first row for duplicate
    # member-month keys. Some duplicate groups conflict, so we retain the
    # source order and disclose the ambiguity instead of summing possible copies.
    activity = activity_raw.drop_duplicates(key, keep="first").copy()
    activity["period"] = activity["Year"] * 12 + activity["Month"]

    # Prediction cohort: enrolled by the end of 2017 and not cancelled by then.
    enrolled_by_cutoff = (
        (members["Enrollment Year"] < AS_OF_YEAR)
        | ((members["Enrollment Year"] == AS_OF_YEAR) & (members["Enrollment Month"] <= AS_OF_MONTH))
    )
    cancelled_by_cutoff = members["Cancellation Year"].notna() & (
        (members["Cancellation Year"] < AS_OF_YEAR)
        | ((members["Cancellation Year"] == AS_OF_YEAR) & (members["Cancellation Month"] <= AS_OF_MONTH))
    )
    cohort = members.loc[enrolled_by_cutoff & ~cancelled_by_cutoff].copy()
    cohort_ids = set(cohort["Loyalty Number"])

    # Reindex to all 12 months for each baseline member; missing activity rows
    # represent zero activity in this monthly dataset.
    months = pd.MultiIndex.from_product(
        [sorted(cohort_ids), [AS_OF_YEAR], range(1, 13)],
        names=["Loyalty Number", "Year", "Month"],
    ).to_frame(index=False)
    baseline_activity = months.merge(
        activity.loc[activity["Year"].eq(AS_OF_YEAR), key + [
            "Total Flights", "Distance", "Points Accumulated", "Points Redeemed", "Dollar Cost Points Redeemed"
        ]],
        on=key,
        how="left",
    ).fillna({
        "Total Flights": 0,
        "Distance": 0,
        "Points Accumulated": 0,
        "Points Redeemed": 0,
        "Dollar Cost Points Redeemed": 0,
    })
    baseline_activity = baseline_activity.sort_values(key).reset_index(drop=True)
    baseline_activity["quarter"] = ((baseline_activity["Month"] - 1) // 3 + 1).astype(int)
    baseline_activity["half"] = np.where(baseline_activity["Month"] <= 6, "H1", "H2")

    activity_features = baseline_activity.groupby("Loyalty Number").agg(
        total_flights_2017=("Total Flights", "sum"),
        active_months_2017=("Total Flights", lambda x: int((x > 0).sum())),
        mean_monthly_flights_2017=("Total Flights", "mean"),
        std_monthly_flights_2017=("Total Flights", "std"),
        total_distance_2017=("Distance", "sum"),
        points_accumulated_2017=("Points Accumulated", "sum"),
        points_redeemed_2017=("Points Redeemed", "sum"),
        dollar_cost_redeemed_2017=("Dollar Cost Points Redeemed", "sum"),
    ).reset_index()
    activity_features["std_monthly_flights_2017"] = activity_features["std_monthly_flights_2017"].fillna(0)

    quarterly = baseline_activity.pivot_table(
        index="Loyalty Number", columns="quarter", values="Total Flights", aggfunc="sum", fill_value=0
    ).reindex(columns=range(1, 5), fill_value=0)
    quarterly.columns = [f"q{q}_flights_2017" for q in quarterly.columns]
    activity_features = activity_features.merge(quarterly.reset_index(), on="Loyalty Number", how="left")

    last_flight = (
        baseline_activity.loc[baseline_activity["Total Flights"] > 0]
        .groupby("Loyalty Number")["Month"].max()
    )
    activity_features["months_since_last_flight"] = (
        activity_features["Loyalty Number"].map(last_flight).map(
            lambda m: _months_since(AS_OF_YEAR, int(m), AS_OF_YEAR, AS_OF_MONTH) if pd.notna(m) else 12
        )
    )
    zero_runs = baseline_activity.groupby("Loyalty Number")["Total Flights"].apply(
        lambda x: _longest_zero_run((x.to_numpy() > 0))
    )
    activity_features["longest_zero_run_2017"] = activity_features["Loyalty Number"].map(zero_runs)
    activity_features["flights_h1_2017"] = activity_features[[f"q{q}_flights_2017" for q in (1, 2)]].sum(axis=1)
    activity_features["flights_h2_2017"] = activity_features[[f"q{q}_flights_2017" for q in (3, 4)]].sum(axis=1)
    activity_features["flight_trend_h2_vs_h1_2017"] = (
        activity_features["flights_h2_2017"] - activity_features["flights_h1_2017"]
    )
    activity_features["redemption_rate_2017"] = np.where(
        activity_features["points_accumulated_2017"] > 0,
        activity_features["points_redeemed_2017"] / activity_features["points_accumulated_2017"],
        0,
    )
    activity_features["tenure_months_asof_2017_12"] = (
        (AS_OF_YEAR - cohort.set_index("Loyalty Number")["Enrollment Year"]) * 12
        + (AS_OF_MONTH - cohort.set_index("Loyalty Number")["Enrollment Month"])
    ).reindex(activity_features["Loyalty Number"]).to_numpy()

    # Future label uses 2018 only. 2018 member activity and the supplied
    # snapshot CLV/card tier are never predictors.
    outcome_activity = activity.loc[activity["Year"].eq(2018) & activity["Loyalty Number"].isin(cohort_ids)]
    outcome_months = pd.MultiIndex.from_product(
        [sorted(cohort_ids), [2018], range(1, 13)],
        names=["Loyalty Number", "Year", "Month"],
    ).to_frame(index=False).merge(
        outcome_activity[key + ["Total Flights"]], on=key, how="left"
    )
    outcome_months["Total Flights"] = outcome_months["Total Flights"].fillna(0)
    outcome_months = outcome_months.sort_values(key)
    silent_2018 = outcome_months.groupby("Loyalty Number")["Total Flights"].apply(
        lambda x: _longest_zero_run((x.to_numpy() > 0)) >= 6
    )
    outcomes = cohort.set_index("Loyalty Number")
    formally_churned_2018 = outcomes["Cancellation Year"].eq(2018)
    outcomes = pd.DataFrame({
        "Loyalty Number": outcomes.index,
        "formally_churned_2018": formally_churned_2018.astype(int).to_numpy(),
        "silent_churned_2018": (
            ~formally_churned_2018 & silent_2018.reindex(outcomes.index).fillna(False)
        ).astype(int).to_numpy(),
    })
    outcomes["broad_churn_2018"] = (
        outcomes["formally_churned_2018"] | outcomes["silent_churned_2018"]
    ).astype(int)

    # Sensitivity to source ordering for duplicate monthly keys. The activity
    # rows have no field that identifies the authoritative duplicate.
    last_version = activity_raw.drop_duplicates(key, keep="last")
    last_outcome = last_version.loc[
        last_version["Year"].eq(2018) & last_version["Loyalty Number"].isin(cohort_ids),
        key + ["Total Flights"],
    ]
    last_grid = months.assign(Year=2018).merge(last_outcome, on=key, how="left")
    last_grid["Total Flights"] = last_grid["Total Flights"].fillna(0)
    last_grid = last_grid.sort_values(key)
    last_silent = last_grid.groupby("Loyalty Number")["Total Flights"].apply(
        lambda x: _longest_zero_run(x.to_numpy() > 0) >= 6
    )
    last_broad = (
        formally_churned_2018
        | (~formally_churned_2018 & last_silent.reindex(formally_churned_2018.index).fillna(False))
    ).astype(int)
    first_broad = outcomes.set_index("Loyalty Number")["broad_churn_2018"]
    first_2017_flights = (
        activity.loc[activity["Year"].eq(2017) & activity["Loyalty Number"].isin(cohort_ids)]
        .groupby("Loyalty Number")["Total Flights"].sum()
        .reindex(sorted(cohort_ids), fill_value=0)
    )
    last_2017_flights = (
        last_version.loc[last_version["Year"].eq(2017) & last_version["Loyalty Number"].isin(cohort_ids)]
        .groupby("Loyalty Number")["Total Flights"].sum()
        .reindex(sorted(cohort_ids), fill_value=0)
    )
    duplicate_policy_sensitivity = {
        "first_row_broad_churners": int(first_broad.sum()),
        "last_row_broad_churners": int(last_broad.sum()),
        "members_with_changed_broad_churn_label": int((first_broad != last_broad).sum()),
        "first_row_rate": float(first_broad.mean()),
        "last_row_rate": float(last_broad.mean()),
        "members_with_changed_2017_flight_total": int(first_2017_flights.ne(last_2017_flights).sum()),
        "first_row_total_2017_flights": int(first_2017_flights.sum()),
        "last_row_total_2017_flights": int(last_2017_flights.sum()),
    }

    features = cohort[["Loyalty Number", "Province", "Enrollment Year", "Enrollment Month"]].merge(
        activity_features, on="Loyalty Number", how="left"
    ).merge(outcomes, on="Loyalty Number", how="left")

    # A/B reference cohort: at least one 2017 flight and >=3 full months since
    # the last flight at the 2017 cutoff. 2018 Q1 activity is only a historical
    # benchmark for experiment sizing; no randomized assignment exists.
    experiment_cohort = features.loc[
        features["total_flights_2017"].gt(0) & features["months_since_last_flight"].ge(3),
        ["Loyalty Number", "months_since_last_flight", "total_flights_2017"],
    ].copy()
    q1_2018 = (
        activity.loc[activity["Year"].eq(2018) & activity["Month"].between(1, 3)]
        .groupby("Loyalty Number")["Total Flights"].sum()
    )
    experiment_cohort["flight_within_90d"] = (
        experiment_cohort["Loyalty Number"].map(q1_2018).fillna(0).gt(0).astype(int)
    )

    audit = {
        "members_in_source": int(len(members)),
        "activity_rows_raw": int(len(activity_raw)),
        "activity_rows_after_member_month_dedup": int(len(activity)),
        "duplicate_member_month_rows_removed": duplicate_rows,
        "duplicate_member_month_groups": duplicate_groups,
        "conflicting_duplicate_groups": conflicting_duplicate_groups,
        "duplicate_policy_sensitivity_first_vs_last": duplicate_policy_sensitivity,
        "eligible_cohort_size": int(len(features)),
        "formal_2018_cancellations": int(features["formally_churned_2018"].sum()),
        "silent_2018_churners": int(features["silent_churned_2018"].sum()),
        "broad_2018_churners": int(features["broad_churn_2018"].sum()),
        "broad_2018_churn_rate": float(features["broad_churn_2018"].mean()),
        "experiment_historical_at_risk_n": int(len(experiment_cohort)),
        "experiment_historical_90d_returners": int(experiment_cohort["flight_within_90d"].sum()),
        "experiment_historical_90d_return_rate": float(experiment_cohort["flight_within_90d"].mean()) if len(experiment_cohort) else None,
    }
    return features, experiment_cohort, audit


def _two_proportion_n_per_arm(p0: float, p1: float, alpha: float = 0.05, power: float = 0.80) -> int:
    if p1 <= p0 or p1 >= 1 or p0 <= 0 or p0 >= 1:
        return 0
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    pooled = (p0 + p1) / 2
    numerator = z_alpha * math.sqrt(2 * pooled * (1 - pooled))
    numerator += z_power * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
    return math.ceil(numerator**2 / (p1 - p0) ** 2)


def _build_experiment_power(experiment_cohort: pd.DataFrame) -> pd.DataFrame:
    baseline = float(experiment_cohort["flight_within_90d"].mean()) if len(experiment_cohort) else 0.0
    rows = []
    for lift in (0.03, 0.05, 0.08, 0.10):
        treatment_rate = baseline + lift
        n = _two_proportion_n_per_arm(baseline, treatment_rate)
        inflated_n = math.ceil(n / 0.90) if n else 0  # 10% unusable follow-up allowance
        rows.append({
            "control_90d_return_rate": baseline,
            "absolute_lift": lift,
            "target_treatment_rate": treatment_rate if treatment_rate < 1 else None,
            "alpha_two_sided": 0.05,
            "power": 0.80,
            "n_per_arm_complete": n if n else None,
            "n_per_arm_with_10pct_allowance": inflated_n if inflated_n else None,
            "total_randomized_with_allowance": 2 * inflated_n if inflated_n else None,
            "fits_historical_analogue_sample_size": bool(2 * inflated_n <= len(experiment_cohort)) if inflated_n else False,
        })
    return pd.DataFrame(rows)


def _model_candidates() -> dict:
    return {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        ),
        "Histogram Gradient Boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=180,
                learning_rate=0.07,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            ),
        ),
    }


def _score_holdout(y_true: pd.Series, probabilities: np.ndarray) -> dict:
    y_pred = (probabilities >= 0.5).astype(int)
    top_n = max(1, int(math.ceil(len(y_true) * 0.10)))
    top_idx = np.argsort(probabilities)[-top_n:]
    base_rate = float(y_true.mean())
    top_rate = float(y_true.iloc[top_idx].mean())
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "positive_rate": base_rate,
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "precision_at_0_5": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix_labels_0_1": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "top_decile_n": top_n,
        "top_decile_positive_rate": top_rate,
        "top_decile_lift_over_base": float(top_rate / base_rate) if base_rate else None,
        "top_decile_recall": float(y_true.iloc[top_idx].sum() / y_true.sum()) if y_true.sum() else None,
    }


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    features, experiment_cohort, audit = build_cohort_and_features()
    target = "broad_churn_2018"
    id_columns = ["Loyalty Number", "Province", "Enrollment Year", "Enrollment Month"]
    feature_columns = [
        c for c in features.columns
        if c not in id_columns and c not in {"formally_churned_2018", "silent_churned_2018", "broad_churn_2018"}
    ]
    X = features[feature_columns].astype(float)
    y = features[target].astype(int)
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, features.index, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    candidates = _model_candidates()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_average_precision = {}
    for name, estimator in candidates.items():
        scores = cross_val_score(estimator, X_train, y_train, scoring="average_precision", cv=cv, n_jobs=-1)
        cv_average_precision[name] = {
            "fold_average_precision": [float(x) for x in scores],
            "mean_average_precision": float(scores.mean()),
            "std_average_precision": float(scores.std(ddof=1)),
        }

    selected_name = max(cv_average_precision, key=lambda k: cv_average_precision[k]["mean_average_precision"])
    selected = candidates[selected_name]
    calibrated = CalibratedClassifierCV(estimator=selected, method="sigmoid", cv=5, n_jobs=-1)
    calibrated.fit(X_train, y_train)
    holdout_probabilities = calibrated.predict_proba(X_test)[:, 1]
    holdout_metrics = _score_holdout(y_test.reset_index(drop=True), holdout_probabilities)

    # Cross-validated scores keep the sample-level dashboard from displaying
    # in-sample fitted risk values. They remain historical scores, not live risk.
    selected_for_oof = CalibratedClassifierCV(estimator=selected, method="sigmoid", cv=3, n_jobs=-1)
    oof_probabilities = cross_val_predict(
        selected_for_oof,
        X,
        y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 1),
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    scored = features[id_columns + [
        "total_flights_2017", "active_months_2017", "total_distance_2017",
        "months_since_last_flight", "flight_trend_h2_vs_h1_2017",
        "formally_churned_2018", "silent_churned_2018", "broad_churn_2018",
    ]].copy()
    scored["churn_probability"] = oof_probabilities
    scored["churn_risk_tier"] = pd.qcut(
        scored["churn_probability"].rank(method="first"),
        q=[0, 0.6, 0.85, 1],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    q75 = features["total_flights_2017"].quantile(0.75)
    high_activity = features["total_flights_2017"] >= q75
    inactive = features["months_since_last_flight"] >= 3
    trend = features["flight_trend_h2_vs_h1_2017"]
    scored["segment"] = np.select(
        [
            (features["total_flights_2017"] == 0),
            high_activity & inactive,
            (trend >= 5) & ~inactive,
            (trend <= -5) & ~inactive,
            ~inactive,
        ],
        ["No 2017 Activity", "High Activity, Currently Inactive", "Growing Activity", "Declining Activity", "Stable Active"],
        default="Low Activity / Inactive",
    )
    # Risk score × prior flying volume is a transparent outreach ordering proxy,
    # not a revenue or profit estimate.
    scored["priority_score"] = scored["churn_probability"] * np.log1p(scored["total_flights_2017"])
    scored = scored.rename(columns={
        "broad_churn_2018": "broad_churn",
        "formally_churned_2018": "formally_churned",
    })
    scored.to_csv(PROCESSED_DIR / "members_segmented.csv", index=False)

    power_table = _build_experiment_power(experiment_cohort)
    power_table.to_csv(RESULTS_DIR / "retention_ab_power.csv", index=False)
    experiment_cohort.to_csv(RESULTS_DIR / "historical_experiment_baseline.csv", index=False)

    report = {
        "problem_framing": "Predict broad 2018 disengagement using only activity observed through 2017-12-31.",
        "feature_cutoff": "2017-12-31",
        "outcome_window": "2018-01-01 through 2018-12-31",
        "target_definition": "2018 cancellation OR at least 6 consecutive zero-flight months during 2018 (among members active in the program on 2017-12-31).",
        "predictor_features": feature_columns,
        "selected_model": selected_name,
        "model_selection_metric": "Mean five-fold average precision on the training split; test split is held out until final evaluation.",
        "cross_validation": cv_average_precision,
        "holdout_metrics": holdout_metrics,
        "data_quality_audit": audit,
        "ab_test_design": {
            "randomized_assignment_available": False,
            "historical_assignment": "None in the supplied files; no treatment effect or causal lift is estimated.",
            "eligibility": "Enrolled and uncancelled at 2017-12-31, at least one flight during 2017, and at least 3 months since the last 2017 flight.",
            "historical_control_proxy": "Observed any-flight rate in Jan-Mar 2018 for the eligible historical cohort. This is a baseline reference, not a randomized control arm.",
            "primary_outcome_for_proposed_trial": "At least one flight during the 90 days after randomized assignment (intent-to-treat).",
            "treatment_example": "Personalized route reminder plus a pre-specified points offer versus business-as-usual communication.",
            "randomization_unit": "Loyalty Number, stratified by predicted risk tier.",
            "power_assumptions": "Two-sided two-proportion normal approximation, alpha=0.05, power=0.80, equal allocation, 10% allowance for unusable follow-up.",
        },
    }
    (RESULTS_DIR / "temporal_model_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({
        "cohort_size": audit["eligible_cohort_size"],
        "2018_broad_churn_rate": audit["broad_2018_churn_rate"],
        "selected_model": selected_name,
        "holdout_metrics": holdout_metrics,
        "experiment_baseline_n": audit["experiment_historical_at_risk_n"],
        "experiment_90d_return_rate": audit["experiment_historical_90d_return_rate"],
        "outputs": [
            str(PROCESSED_DIR / "members_segmented.csv"),
            str(RESULTS_DIR / "temporal_model_metrics.json"),
            str(RESULTS_DIR / "retention_ab_power.csv"),
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
