# Leakage-Aware Airline Churn Model Report

## Problem and prediction time

The original submission identified disengagement using the full Jan 2017–Dec 2018 activity period, then trained a classifier on features from that same period. Because several features directly described the label, its very high AUC was not a valid estimate of future prediction quality.

This re-analysis fixes the prediction date at **31 December 2017**. The eligible cohort includes members enrolled by that date and not formally cancelled by it. Only 2017 activity and enrollment tenure are used as predictors. The target is observed during calendar 2018:

- **Formal churn:** cancellation recorded during 2018.
- **Silent churn:** no formal 2018 cancellation and at least six consecutive zero-flight months during 2018.
- **Broad churn:** either formal or silent churn.

This is a retrospective forecast experiment over one year, not a live model and not a retention impact study.

## Preparation and features

The input includes 16,737 member records and 392,936 monthly activity rows. The pipeline creates a 12-month grid for each eligible member and fills absent activity rows with zero. Predictor features are 2017 flight counts, active months, distance and points totals, redemption rate, quarterly activity, H1-to-H2 activity change, longest zero-flight run, months since the last flight, and enrollment tenure as of the cutoff.

The model excludes 2018 activity, the cancellation fields, the supplied snapshot CLV, and card tier. Thus, no part of the 2018 target or current member-value snapshot enters the predictors.

There are 3,871 duplicate member-month rows across 3,847 keys. Some duplicate rows conflict; 1,943 duplicate groups differ in at least one activity value. The analysis retains the first source row per member-month to match the original pipeline. Keeping the last row changes the 2017 total-flight feature for 94 members (aggregate flights: 212,759 vs. 211,091) and changes the broad-churn count from 966 to 990 (24 labels; 7.85% to 8.05%). This ambiguity should be resolved with the source owner before operational use.

## Cohort outcome

| Measure | Members | Rate |
|---|---:|---:|
| Eligible at 2017-12-31 | 12,305 | 100.00% |
| Formal cancellations in 2018 | 486 | 3.95% |
| Silent churners in 2018 | 480 | 3.90% |
| Broad churn in 2018 | 966 | 7.85% |

## Model selection and holdout evaluation

The data is split by member into 80% training and 20% holdout sets, stratified by the 2018 target. Five-fold average-precision cross-validation on the training split compares Logistic Regression and Histogram Gradient Boosting. Histogram Gradient Boosting was selected (mean average precision 0.436 vs 0.424 for Logistic Regression); the difference is modest relative to fold variability. A sigmoid calibrator is fit within the training split before the final holdout evaluation.

| Holdout metric | Result |
|---|---:|
| Members | 2,461 |
| 2018 broad-churn prevalence | 7.84% |
| ROC-AUC | 0.729 |
| Average precision | 0.447 |
| Brier score | 0.0527 |
| Top-decile churn rate | 33.2% |
| Top-decile lift over holdout prevalence | 4.23× |
| Share of all holdout churners in top decile | 42.5% |

At a fixed 0.50 cutoff, holdout precision is 96.6% and recall is 29.0%. That cutoff is reported for reference only; no intervention costs or value data were provided to choose an operational threshold. The risk decile is a ranking aid, not a complete contact policy.

## Interpretation

Using only the prior-year activity, the model ranks members with meaningfully different 2018 outcomes: one in three members in the highest-score decile met the broad-churn label, compared with about one in thirteen members overall. This supports prioritizing a test cohort, but does not show that contacting them or offering bonus points would change their behavior.

## Limits

- The holdout split is across members within a single historical forecast year. There is no later year to test temporal transport.
- The silent-churn rule is a chosen operational proxy, not a ground-truth statement that a member permanently left.
- Duplicate member-month rows conflict and the source lacks an authoritative correction field.
- Flight activity does not include campaign assignment, offer exposure, incentive cost, or contribution margin. No treatment effect or ROI can be estimated.
- Enrollment/demographic records are treated as static; the data does not provide their historical update dates.

Detailed metrics and the duplicate audit are saved in `results/temporal_model_metrics.json` after running the pipeline.
