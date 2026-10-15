# Airline Loyalty Analytics: Technical Report

> **Legacy submission report — not the current evaluation.** Its model uses full 2017–2018 activity to predict a label built from that same period, causing target leakage. The reported AUC and churn rates are retained for provenance only. Use `model_training_report.md` and `src/temporal_retention_analysis.py` for the leakage-aware re-analysis.

**Prepared by:** Business Analytics Team  
**Dataset:** 16,737 Canadian Loyalty Members | 2012–2018 Enrollment | 2017–2018 Flight Activity

---

## Section 1: Problem Framing

### Background

Airlines invest heavily in loyalty programs to generate repeat business and grow Customer Lifetime Value (CLV). The standard management view of these programs focuses on two things: points accumulation and formal cancellations. However, this framing misses the most expensive problem — members who quietly stop flying without ever cancelling.

The marketing team's current dashboard reports a **12.3% formal cancellation rate**. This number feels manageable. The reality hidden beneath it is that an additional **22.9% of members have been completely inactive for 6 or more consecutive months** without cancelling. They are still listed as "active" in the CRM, still receiving marketing emails, and still consuming budget — while generating zero revenue.

### What We Set Out to Solve

The goal of this project was threefold:

1. **Redefine churn** to capture the full scope of disengagement, not just formal cancellations
2. **Segment the member base** into groups that are meaningfully distinct in their behavior and require different interventions
3. **Build a dashboard** that a non-technical marketing manager can open, understand in minutes, and immediately act on

### The Data Constraint

Flight activity data covers only **January 2017 through December 2018** (24 months). Enrollment records go back to 2012, but behavioral signals — how often someone flies, whether they are trending up or down, how long they have been silent — can only be measured within this 24-month window. CLV (Customer Lifetime Value) is the only bridge to pre-2017 history, as it captures the total invoice value since enrollment.

This constraint is documented and accepted. It means the model's behavioral features are grounded in recent, observable behavior — which is actually what matters most for a retention intervention. A member who flew 20 times in 2014 but hasn't flown since 2017 is disengaged, regardless of their historical CLV.

---

## Section 2: Data Cleaning Decisions

The dataset was real-world in character: it contained anomalies, inconsistencies, and structural quirks that required explicit decisions before any analysis could begin. Every decision is documented below with its justification.

### Issue 1: Negative Salary Values (20 members)

**Finding:** 20 members had negative salary values, with a minimum of −$58,486. These cannot represent real incomes.

**Decision:** Treat as data entry errors. Set to null. These 20 rows are absorbed into Issue 2 below.

**Justification:** No plausible business interpretation exists for a negative salary. Converting to null (rather than zero or median) preserves the integrity of the imputation step.

---

### Issue 2: Salary Nulls (4,258 members = 25.4%)

**Finding:** 4,238 members had missing salary data in the original file, plus the 20 negatives from Issue 1 = **4,258 total nulls** (25.4% of all members).

**Decision:** Impute using grouped median: first by (Province × Education × Loyalty Card tier), then fall back to Province median, then global median. A binary flag column `salary_imputed = 1` marks every imputed row.

**Justification:** A simple global median would ignore real salary variation by region and education level. Grouped medians respect these gradients. The flag preserves analytical honesty — salary-based insights are treated as indicative, not precise, for imputed members.

---

### Issue 3: Duplicate Flight Rows (158 members, 7,718 excess rows)

**Finding:** 158 members had 2–3× the expected number of flight activity rows (up to 72 rows per member instead of 24). These were exact duplicates of (Loyalty Number, Year, Month) combinations — a data entry or system export error.

**Decision:** Deduplicate strictly on `[Loyalty Number, Year, Month]`, keeping the first occurrence.

**Justification:** Keeping duplicates would double or triple flight counts for 158 members, making their behavioral features severely inflated and unrepresentative. The `keep='first'` strategy is conservative and consistent.

---

### Issue 4: 2018 Promotion Members with Partial Records (971 members)

**Finding:** 971 members enrolled under a "2018 Promotion" and consequently have flight records covering only part of the observation window. The 24-month grid fills their 2017 months with zero flights — because they hadn't enrolled yet.

**Decision:** Exclude these 971 members from the **silent-churn rule**. For them, only formal cancellations count as churn. Their behavioral features are flagged with `is_promo_member = 1`.

**Justification:** Without this exclusion, all 971 promotion members would have `max_consec_zeros ≥ 13` (12 structural zeros in 2017 plus variable 2018 activity), automatically triggering the 6-month inactivity threshold and labelling 100% of them as silent churners. This is analytically incorrect — their 2017 zeros reflect non-enrollment, not disengagement. Of the 971 promotion members, 115 (11.8%) formally cancelled, which is their correct churn count.

---

### Issue 5: Building the Complete 24-Month Activity Grid

**Finding:** After deduplication, 49 members had fewer than 24 rows in the flight activity file. The 971 promotion members had only 11 rows each.

**Decision:** Cross-join all 16,737 members with all 24 (Year, Month) combinations to produce a complete 401,688-row grid. Any (member, month) pair with no matching flight record is filled with zeros across all flight columns.

**Justification:** A complete grid ensures every member is evaluated on the same 24-month scale. Without it, members with missing months would have systematically underrepresented behavioral features (e.g., lower total flights simply because rows were missing).

---

### Data Quality Summary Table

| Issue | Scope | Decision | Flag Added |
|---|---|---|---|
| Negative salaries | 20 members | Nullify → impute | `salary_imputed` |
| Salary nulls | 4,238 members (25.3%) | Grouped median imputation | `salary_imputed` |
| Duplicate flight rows | 158 members, 7,718 rows | Deduplicate on (Member, Year, Month) | None |
| Promo partial records | 971 members, 11 rows each | Grid-fill with 0; exclude silent-churn rule | `is_promo_member` |
| Incomplete grid rows | 49 members | Fill missing months with 0 | None |

---

## Section 3: Churn Model

### Redefining Churn

The standard cancellation-only definition identifies **2,067 churned members (12.3%)**. Treating this as the full picture misses the more costly problem.

We define **broad churn** as:

> **Broad churn = 1** if: (a) the member formally cancelled, OR (b) the member had 6 or more consecutive zero-flight months and has not formally cancelled.

This yields **5,905 churned members — 35.3% of the member base**. The gap between 12.3% and 35.3% represents **3,838 members** (22.9%) who are silently disengaged. They appear "active" in standard reporting but have generated zero flight revenue for 6+ months.

### Feature Engineering

We constructed 42 features across four categories:

**Recency (strongest predictors):**
- `months_since_last_flight` — churned members average **8.73 months** since last flight vs **0.74 months** for active members
- `max_consec_zeros` — the maximum streak of zero-flight months, which directly encodes the silent-churn signal

**Volume:**
- `total_flights_24m` — churned members average **13.5 flights** vs **39.5 for active members** over 24 months
- `active_months`, `avg_monthly_flights`, `total_distance`

**Trend:**
- `flight_trend` = flights_2018 − flights_2017 — a negative trend is a strong early warning signal
- `seasonal_concentration` — how concentrated a member's flying is in a single quarter

**Value & Engagement:**
- `total_points_accumulated`, `total_points_redeemed`, `redemption_rate`
- Demographics: Province, Education level, Salary (imputed), Loyalty Card tier, Tenure

### Model Selection

Three models were trained on an 80% training / 20% hold-out test split, stratified on `broad_churn`:

| Model | Purpose | Notes |
|---|---|---|
| Logistic Regression | Linear baseline; interpretable coefficients | Required StandardScaler |
| Random Forest | Non-linear relationships; feature importance | `class_weight='balanced'` |
| **XGBoost** | **Primary model** | `scale_pos_weight` dynamically set |

**XGBoost was selected** as the primary model for its superior handling of tabular data, iterative error correction, and ability to capture non-linear interactions between behavioral features. Class imbalance was addressed by dynamically setting `scale_pos_weight = count(active) / count(churned)` from the training data.

### Model Performance

| Metric | Class 0 (Active) | Class 1 (Churned) |
|---|---|---|
| Precision | 0.99 | 1.00 |
| Recall | 1.00 | 0.98 |
| F1-Score | 0.99 | 0.99 |

**ROC-AUC: 0.9969**

The model assigns a continuous churn probability (0–100%) to every member. This score — not just the binary prediction — drives all downstream prioritization.

**Note on model performance:** The near-perfect AUC reflects that `max_consec_zeros` and `months_since_last_flight` are both features in the model and conceptually overlap with the churn label definition. This is by design: the model is a **retrospective behavioral classifier** that scores who is disengaged *today*, not a forward predictor of future behaviour. Its production use case is monthly re-scoring of all active members to prioritise outreach. For this purpose, using full behavioral history is appropriate.

---

## Section 4: Customer Segmentation

### Why CLV Alone Is Not Enough

The program already tracks Customer Lifetime Value. The tempting assumption is that high-CLV members are "safe" and low-CLV members are expendable. The data refutes this:

> **Average CLV of churned members: $8,048 vs. average CLV of active members: $7,957.**

The difference is $91 — statistically negligible. **CLV does not predict churn.** Members are not disengaging because they were low-value; they are disengaging silently regardless of their historical spend. This means the program is currently unable to distinguish a member who will be worth $15,000 over the next five years from one who has already mentally left.

Behavioral signals — consecutive months of inactivity, year-over-year flight trend — are the only reliable early-warning indicators.

### The 7-Segment Framework

We combined CLV (value dimension) and behavioral signals (engagement dimension) into a 7-segment framework. Thresholds are derived directly from the data distribution:

- **High CLV floor (P75):** $8,940.58
- **Mid CLV floor (P25):** $3,980.84
- **Median flights (24m):** 34.0

| Segment | Size | % | Avg CLV | Avg Flights (24m) | Avg Months Inactive |
|---|---|---|---|---|---|
| **Champions** | 1,928 | 11.5% | $16,594 | 43.9 | 0.4 |
| **At-Risk High Value** | 89 | 0.5% | $16,163 | 41.7 | 3.4 |
| **Rising Members** | 1,694 | 10.1% | $7,072 | 40.7 | 0.5 |
| **Stable Regulars** | 2,922 | 17.5% | $7,512 | 37.0 | 0.6 |
| **Declining Regulars** | 1,437 | 8.6% | $7,354 | 38.6 | 1.0 |
| **Low Engagement** | 2,762 | 16.5% | $2,990 | 38.6 | 1.1 |
| **Churned / Dormant** | 5,905 | 35.3% | $8,048 | 13.5 | 8.7 |

**Segment definitions:**
- **Champions:** CLV ≥ P75 AND flights ≥ median AND not at risk
- **At-Risk High Value:** CLV ≥ P75 AND flights ≥ median AND 3+ months inactive
- **Rising Members:** CLV ≥ P25 AND positive flight trend (>5 flights 2018 vs 2017)
- **Declining Regulars:** CLV ≥ P25 AND negative flight trend (<−5 flights)
- **Stable Regulars:** CLV ≥ P25 AND flat trend AND active
- **Low Engagement:** Active but CLV < P25 or flights < threshold
- **Churned / Dormant:** broad_churn = 1 (formal cancel OR 6m+ consecutive inactive)

### Retention Strategy Per Segment

Each segment has a specific intervention mapped to a trigger, offer, and timing:

| Segment | Trigger | Offer | Timing |
|---|---|---|---|
| **Champions** | Flight frequency drop >20% vs prior quarter | Lounge access renewal + preview event invite | 30 days before tier anniversary |
| **At-Risk High Value** | 3 consecutive months without a flight | Direct call from Senior Loyalty Team; 2× points next 3 flights + waived rebooking (90 days) | Within 5 business days of trigger |
| **Rising Members** | >5-flight increase in rolling 6-month window | "X flights from Aurora" progress tracker; tier milestone bonus | When member reaches 80% of next-tier threshold |
| **Declining Regulars** | 2 consecutive months below prior 6-month average | 10–15% fare discount + double points on top 3 routes | Q2 launch (April — before peak summer) |
| **Stable Regulars** | Seasonal (routine, not behavioral) | Points bonus on bookings made 45+ days in advance | March 1 and October 1 each year |
| **Churned / Dormant** | 4 months inactive (Wave 1) or cancellation record (Wave 2) | Wave 1: "We miss you" + 5,000 bonus points; Wave 2: Match old tier + 10,000 points | Month 4 (Wave 1) or within 30 days of cancel (Wave 2) |
| **Low Engagement** | Enrolled 6m+, fewer than 2 flights in last 6 months | Automated 3-touch email: inspiration → points expiry reminder → seasonal deal | Monthly automated batch — zero direct spend |

---

## Section 5: Key Findings

### Finding 1 — The Silent Leakage Problem

> *"3,838 members are enrolled but have not flown for 6 or more consecutive months without formally cancelling. At their average CLV of $8,002, this group represents over **$30.7 million in at-risk lifetime value**. At a conservative 20% reactivation rate, that is **$6.1 million in recoverable revenue** — currently invisible in the programme's standard cancellation-only churn reports."*

This is the single most important finding. The programme's current KPI (formal churn rate: 12.3%) is not wrong — but it is **incomplete**. The real disengagement rate is 35.3%, and two-thirds of that is happening silently.

These members are not lost — they are simply not being noticed. Unlike formally cancelled members, their contact details are valid, their accounts are open, and they can be reactivated. The window is finite: after 18 months of silence, reactivation rates drop precipitously and further marketing spend becomes wasteful.

---

### Finding 2 — First-Year Churn Is the Dominant Failure Mode

> *"1,509 of 2,067 formally churned members — **73%** — cancelled within 12 months of enrollment. The programme is losing nearly three-quarters of its formally-churned customers in the critical first year, before most loyalty behaviours have had time to form."*

This finding changes where intervention resources should be directed. The common instinct is to focus retention budgets on long-tenure members at risk of leaving. But the data shows the greatest attrition happens before members even become loyal. A first-flight reward, a 3-month milestone email, and a 6-month activity check-in would cost a fraction of a win-back campaign — and address the problem before it becomes one.

---

### Finding 3 — Flight Trend Beats Points Balance as an Early Warning Signal

> *"The top predictors of churn are consecutive months of inactivity and year-over-year flight trend. A member's points balance, CLV, and card tier have negligible predictive power in comparison. The programme currently has no automated alert for members showing a declining flight trajectory. Implementing one is the highest-ROI operational change available."*

This is counterintuitive to a team that manages a points-based loyalty programme. The data shows that **what members do** (fly frequency, trend, recency) matters far more than **what they have** (points balance, CLV, tier). A member with 50,000 points who hasn't flown in 4 months is far more at risk than a member with 5,000 points who flew last month. The current system has no way to see this.

---

## Section 6: Three Business Recommendations

### Recommendation 1 — Launch an Early-Lifecycle Retention Programme (First 12 Months)

**The problem:** 73% of formally churned members leave within year one. The programme has no structured onboarding or early-lifecycle engagement.

**The recommendation:** Design a 3-touch automated sequence for all new members:
- **Month 1:** Welcome email with clear explanation of how points work and what Aurora status means
- **Month 3 (or first flight, whichever comes first):** First-flight milestone reward — 1,000 bonus points
- **Month 6:** If no flight has occurred, trigger the "We miss you" campaign before the member is fully disengaged

**Who receives it:** All new enrollees from the first day of enrolment.

**Expected impact:** If early-lifecycle interventions reduce Year-1 formal churn by 30% (a conservative estimate from industry benchmarks), that converts ~450 members per year from churned to active — each with an average CLV of $8,000, representing $3.6M in protected lifetime value annually.

**Why a CFO can support it:** The cost is email automation — near-zero marginal cost per member. The ROI calculation is straightforward and does not require expensive discount offers.

**Why a CMO can support it:** It directly addresses the programme's worst-performing cohort. First-year engagement is the foundation of brand loyalty. Fixing this converts the programme from a revolving door into a retention engine.

---

### Recommendation 2 — Implement a Flight-Trend Alert System

**The problem:** The top predictors of churn are behavioral signals that change gradually over months. The programme currently has no mechanism to detect a member who is trending toward disengagement before they fully disengage.

**The recommendation:** Build an automated monitoring rule that fires when any member's rolling 3-month average flights drops more than 20% below their prior 6-month average. Route the alert to the retention team for personalised outreach within 5 business days.

**Who receives it:** Any member in the Champions, At-Risk High Value, or Declining Regulars segments whose rolling flight average is declining.

**Expected impact:** The 89 At-Risk High Value members represent an average CLV of $16,163 each — a total of ~$1.4M in near-term risk. Catching even 30% of these members before they fully disengage would prevent significant value loss. Across Declining Regulars (1,437 members, avg CLV $7,354), the opportunity is proportionally larger.

**Why a CFO can support it:** This is the highest-ROI intervention available because it targets high-value members *before* they churn, when win-back costs are lowest and success rates are highest.

**Why a CMO can support it:** Personalised outreach to a member who is showing early signs of disengagement is perceived as attentive, not intrusive. It builds trust rather than eroding it, unlike blanket promotional campaigns.

---

### Recommendation 3 — Implement a Sunset vs. Reactivate Protocol for Dormant Members

**The problem:** The 5,905 Churned/Dormant members (35.3% of the base) are heterogeneous. Some cancelled 3 months ago and are reactivatable; others have been silent for 18+ months and are effectively unrecoverable. Treating them all the same wastes marketing budget on unrecoverable members and underinvests in reactivatable ones.

**The recommendation:** Implement a tiered dormancy protocol based on recency:

| Silence Duration | Action | Channel | Offer |
|---|---|---|---|
| 4–6 months | Wave 1: Reactivation | Email | "We miss you" + 5,000 bonus points on next booking |
| 6–12 months | Wave 2: Win-back | Email + SMS | Match old card tier for 6 months + 10,000 bonus points |
| 12–18 months | Final attempt | Email only | No incentive — last-chance route inspiration |
| 18+ months | Suppress | Remove from active CRM | No further spend |

**Who receives it:** All members in the Churned/Dormant segment, tiered by `months_since_last_flight`.

**Expected impact:** Industry reactivation rates for email campaigns targeting members in the 4–12 month window range from 15–25%. Applying a 20% rate to the 3,838 silent churners yields ~768 reactivated members. At average CLV of $8,002, that is approximately $6.1M in protected value. The suppression rule simultaneously eliminates wasted spend on the unrecoverable segment (estimated 30% reduction in dormant member marketing costs).

**Why a CFO can support it:** This recommendation both increases revenue (reactivation) and reduces cost (suppression). It converts a cost centre (dormant member marketing) into a recoverable asset class.

**Why a CMO can support it:** Suppressing members after 18 months of silence is responsible list hygiene. It improves email deliverability, sender reputation, and campaign performance metrics across the board.

---

## Section 7: Known Limitations

| Limitation | Impact on Analysis |
|---|---|
| Flight data covers only 2017–2018 | Pre-2017 behaviour is unobservable; CLV is the only proxy |
| No route information | Cannot personalise interventions by destination |
| No booking channel data | Cannot assess digital vs. agent engagement |
| Salary imputed for 25.4% of members | Salary-based segment insights are indicative, not precise |
| No competitor data | "Flew less" ≠ "churned to a competitor" — may have reduced travel overall |
| CLV definition opaque | Treated as a composite measure; not reverse-engineered |
| Silent churn threshold (6 months) | A business decision, not a statistical optimum — should be validated against win-back response rates |
