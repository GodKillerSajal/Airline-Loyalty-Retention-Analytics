# Airline Loyalty Retention Analytics

A reproducible airline member-retention project built around flight activity, loyalty history, leakage-aware churn prediction, behavioral segmentation, and a Streamlit review dashboard. The working copy includes the original member and activity data supplied with the project; raw and row-level data are excluded from Git.

## What is included

- Monthly data preparation for 16,737 loyalty members and 392,936 activity rows.
- A temporal model that uses activity through **2017-12-31** to predict observed formal cancellation or 6+ months without flights during **2018**.
- Logistic Regression baseline and Histogram Gradient Boosting comparison, calibrated scoring, held-out metrics, and a cross-validated historical member score file.
- A proposed randomized retention experiment with a historical baseline and sample-size planning.
- A five-page Streamlit dashboard for historical risk review, 2017 activity segments, member lookup, and the proposed experiment.

## Selected evaluation results

The eligible 2017 cutoff cohort contains 12,305 members; 966 (7.85%) meet the 2018 broad-churn definition. On a held-out member split, the selected model achieved **0.729 ROC-AUC** and **0.447 average precision** (the holdout event rate was 0.078). The highest-risk decile contained 33.2% 2018 broad churners, about **4.23×** the overall holdout rate, and captured 42.5% of holdout cases.

These are historical prediction results for one year and one dataset. They are not live airline performance or evidence that a retention intervention works. The original submission's approximately 0.997 AUC used 2017–2018 features that directly encoded its full-window churn label; that score is retired and not used in this evaluation.

## Retention A/B test: design, not treatment result

The supplied files contain no randomized campaign assignment, offer exposure, incentive cost, or margin outcome. The project therefore reports **no treatment effect or causal lift**. Instead, it specifies a future member-level randomized test and calculates sample requirements from a historical 90-day flight-return benchmark. See [the experiment design](report/retention_experiment_design.md) and [the model report](report/model_training_report.md).

## Run locally

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Place `Customer Loyalty History.csv` and `Customer Flight Activity.csv` under `Datasets/`.
3. Build the temporal model scores, evaluation report JSON, and A/B sample-size table:

   ```bash
   python src/temporal_retention_analysis.py
   ```

4. Launch the dashboard:

   ```bash
   streamlit run app/streamlit_app.py
   ```

To analyze results from a future randomized test, export one row per randomized member with columns `loyalty_number`, `arm` (`control` or `treatment`), and `flight_within_90d` (0/1), then run:

```bash
python src/analyze_randomized_trial.py path/to/randomized_results.csv
```

The trial analyzer estimates an intent-to-treat difference in return rate with a 95% interval and two-sided p-value. Its causal interpretation depends on correct random assignment and complete outcome capture.

## Data source

The supplied files match the [Airline Loyalty Program (Canada) dataset used in this Kaggle notebook](https://www.kaggle.com/code/siddharth0935/airline-loyalty-program-01/input). The raw CSVs are not included in Git; obtain them from the source and follow its current terms before redistributing.

## Data and modeling limitations

- The source has duplicate member-month keys; 1,943 duplicate groups conflict on at least one activity value. The pipeline retains the first row in source order, matching the original processing choice. Switching to the last row changes the 2018 broad-churn label for 24 of 12,305 cohort members (7.85% to 8.05%).
- Under the same first-vs-last sensitivity, 94 members' 2017 flight totals differ; aggregate cohort flights differ by 1,668 (212,759 vs. 211,091).
- Only one historical forecast window is available. Member-level holdout performance does not establish year-to-year or airline-to-airline generalization.
- The original full-window model and report are retained under `src/legacy_retrospective/` and `report/legacy_submission_model_training_report.md` for provenance only; do not use their scores for prediction.
- Raw data, personal-level scored outputs, and serialized legacy models are not committed.
