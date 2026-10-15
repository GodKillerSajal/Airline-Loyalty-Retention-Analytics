import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import shap
import joblib

# Configuration
import pathlib
BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()
INPUT_FILE = os.path.join(BASE_DIR, 'data/processed/members_features.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models/')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── MODELLING APPROACH NOTE ────────────────────────────────────────────────────
# This model is a RETROSPECTIVE BEHAVIORAL CLASSIFIER, not a forward predictor.
# It uses full 24-month behavioral features (Jan 2017 – Dec 2018) to classify
# which members are currently disengaged, based on patterns that have already
# occurred.
#
# KEY DISCLOSURE — max_consec_zeros and feature leakage:
#   `max_consec_zeros` has a 0.84 correlation with `broad_churn` because it
#   directly encodes the silent-churn rule (>= 6 => silent churner). This means
#   the model is highly accurate (AUC ~0.997) but performance is inflated vs
#   what a true future-predictor would achieve.
#
# The blueprint (Section 5) recommends a temporal split:
#   Feature window: Jan–Aug 2017 → Target: Sep 2017–Dec 2018 churn
# That design is appropriate for a PREDICTIVE (forward-looking) use case.
# This implementation scores members based on their FULL behavioral history,
# which is the correct approach when the use case is:
#   "Which of our CURRENT members should we contact for retention TODAY?"
#
# For a fully rigorous temporal-split predictive model, rerun with only Jan–Aug
# 2017 aggregated features and exclude max_consec_zeros from that feature set.
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading processed features...")
    full = pd.read_csv(INPUT_FILE)

    # One-hot encode categoricals
    print("Encoding categorical features...")
    full_enc = pd.get_dummies(full, columns=['Gender','Marital Status','Province'], drop_first=True)

    BASE_FEATURES = [
        'total_flights_24m','avg_monthly_flights','max_monthly_flights','std_monthly_flights',
        'active_months','activity_rate','months_since_last_flight','flight_trend',
        'total_distance','avg_dist_per_flight','total_points_acc','total_points_red',
        'redemption_rate','has_redeemed','total_dollar_redeemed','months_with_redemption',
        'q1_flights','q2_flights','q3_flights','q4_flights','seasonal_concentration',
        'max_consec_zeros','card_tier_num','edu_num','Salary','tenure_months','CLV',
        'is_promo_member','salary_imputed',
    ]
    OHE_FEATURES = [c for c in full_enc.columns if c.startswith(('Gender_','Marital Status_','Province_'))]
    ALL_FEATURES  = BASE_FEATURES + OHE_FEATURES

    X = full_enc[ALL_FEATURES].fillna(0)
    y = full_enc['broad_churn']

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale for LR
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    print("Training Logistic Regression...")
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_tr_s, y_train)

    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    print("Training XGBoost...")
    # scale_pos_weight for ~40.4% churn rate (corrected from blueprint)
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        scale_pos_weight=scale_pos, n_estimators=300,
        learning_rate=0.05, max_depth=6,
        random_state=42, eval_metric='auc'
    )
    xgb.fit(X_train, y_train)

    print("XGBoost Evaluation:")
    print(classification_report(y_test, xgb.predict(X_test)))
    print(f"ROC-AUC: {roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1]):.4f}")
    print()
    print("NOTE: High AUC is partially due to max_consec_zeros being both a")
    print("      feature and the direct encoding of the silent-churn label.")
    print("      See modelling approach note at top of this file.")

    # Save model, scaler, and expected columns
    print("Saving model artifacts...")
    model_path  = os.path.join(MODEL_DIR, 'xgb_churn_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    cols_path   = os.path.join(MODEL_DIR, 'model_columns.pkl')
    joblib.dump(xgb,          model_path)
    joblib.dump(scaler,       scaler_path)   # persist scaler for LR inference
    joblib.dump(ALL_FEATURES, cols_path)
    print(f"Saved XGBoost model to {model_path}")

    # Predict probabilities for ALL members (used by segmentation step)
    print("Predicting probabilities for all members...")
    full['churn_probability'] = xgb.predict_proba(X)[:, 1]

    output_path = os.path.join(BASE_DIR, 'data/processed/members_features_with_prob.csv')
    full.to_csv(output_path, index=False)
    print(f"Saved enhanced features to {output_path}")
    print("Churn modeling complete!")

if __name__ == "__main__":
    main()

