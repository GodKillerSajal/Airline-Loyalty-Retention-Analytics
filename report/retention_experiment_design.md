# Proposed Airline Retention A/B Test

## Evidence boundary

The supplied data contains member activity and cancellation fields but **no campaign assignment, offer exposure, experiment arm, offer cost, or margin outcome**. It cannot estimate an A/B treatment effect or causal lift. The historical numbers below are planning references only.

## Historical planning cohort

At the 31 December 2017 cutoff, the design identifies members who had at least one flight during 2017 and had been inactive for at least three months. The supplied data contains **799** such members. **712 (89.1%)** took at least one flight during January–March 2018. This historical return rate is not a randomized control-arm result and may differ from a future airline campaign.

## Proposed experiment

- **Population:** eligible enrolled members with at least three months since their last flight and at least one flight during the prior year.
- **Randomization unit:** loyalty number; assign each member once and preserve assignment across channels.
- **Allocation:** 1:1 treatment vs. business-as-usual control, stratified by the 2017 model risk tier.
- **Treatment example:** personalized route reminder plus a capped points offer. The business-as-usual arm continues current communications and offers.
- **Primary outcome:** any flight within 90 days after assignment, analyzed intent-to-treat.
- **Guardrails:** contribution margin after offer cost, total incentive cost, flight revenue, opt-outs, and complaints.
- **Analysis:** one pre-registered 90-day readout; no outcome-driven exclusions or repeated peeking. Use a difference in proportions with a 95% confidence interval and two-sided test.

## Approximate sample-size scenarios

The calculation uses the historical 89.1% return rate as a planning baseline, two-sided α=0.05, 80% power, equal allocation, a normal approximation for two proportions, and 10% extra randomized members for unusable follow-up.

| Absolute return-rate lift to detect | Target treatment rate | Members per arm including allowance | Total | Fits the 799-member historical analogue? |
|---:|---:|---:|---:|:---:|
| 3 percentage points | 92.1% | 1,648 | 3,296 | No |
| 5 percentage points | 94.1% | 536 | 1,072 | No |
| 8 percentage points | 97.1% | 175 | 350 | Yes |
| 10 percentage points | 99.1% | 96 | 192 | Yes |

An 8–10 point lift is a large, unverified effect assumption, not a forecast. Choose the minimum detectable effect from the economics of the offer, then recalculate using the real eligible population, baseline, allocation, outcome capture, and any stratification/clustering before launch. The historical sample's size does not show the test would succeed.

## Analyze a real trial

Populate `examples/randomized_retention_trial_template.csv` with one row per randomized member after complete follow-up, then run:

```bash
python src/analyze_randomized_trial.py examples/randomized_retention_trial_template.csv
```

The script reports an intent-to-treat return-rate difference, Newcombe 95% confidence interval, and two-sided p-value. It should only be interpreted causally if assignment was randomized as specified and the outcome is captured consistently for both arms.
