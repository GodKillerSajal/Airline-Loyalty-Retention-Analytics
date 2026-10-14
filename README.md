# Driving Retention in Airline Loyalty Programs with Data

**Context & Affiliation**: Consulting & Analytics Club, IIT Guwahati (Summer Projects 2026)  
**Domain**: Airline Analytics, Predictive Churn Modeling, Customer Lifetime Value (CLV), Marketing Strategy  
**Dataset**: 16,737 Canadian airline loyalty members across 2012–2018 (demographic loyalty profiles + monthly flight activity time-series).

---

## 1. Problem Statement
Traditional airline loyalty programs operate on static, retrospective tier metrics, failing to identify "silent churners" (high-value flyers whose booking velocity decays months before formal loyalty tier lapse or cancellation). This project builds an early-warning predictive churn engine and multi-page marketing operations dashboard to safeguard millions in at-risk Customer Lifetime Value (CLV).

---

## 2. Actual Technical & Modeling Execution
- **XGBoost Predictive Churn Model**: Trained and fine-tuned an extreme gradient boosted classifier (XGBoost) on 16,737 members, achieving an exceptional **0.9946 ROC-AUC**. Successfully flagged **3,838 "silent churners"** completely invisible to traditional static reports.
- **Dynamic Tier Segmentation & At-Risk CLV Isolation**: Segmented the member base into 7 behavioral tiers, demonstrating that dynamic flight velocity features dramatically outperformed static loyalty metrics to isolate **$30.7M in at-risk CLV**.
- **Interactive Retention Operations Dashboard**: Deployed a comprehensive 5-page Retention Dashboard equipped with priority watchlists, churn probability deciles, and per-member targeted retention intervention workflows for airline marketing managers.

---

## 3. Real Project Metrics
- **16,737 Airline Members Analyzed**: 2012–2018 monthly time-series logs.
- **0.9946 ROC-AUC**: Achieved by the champion XGBoost churn classification model.
- **3,838 Silent Churners Uncovered**: High-value members with declining flight velocity detected early.
- **$30.7 Million in At-Risk CLV Identified**: Quantified across 7 behavioral membership tiers.
- **5-Page Operations Dashboard**: Delivered with interactive priority watchlists and retention playbooks.

---

## 4. Key Files in this Folder
- `Problem Statements.pdf`: Official IIT Guwahati Consulting & Analytics Club problem statement (Problem 3).

---

## 5. IIT Roorkee Resume Bullets (Strict 1-Line Format)
```text
• **Built XGBoost churn model** (**AUC 0.9946**) on **16,737 members**, uncovering **3,838 silent churners**.
• **Segmented members into 7 tiers**, isolating **$30.7M in at-risk CLV** using dynamic flight activity metrics.
• **Deployed 5-page Retention dashboard** with priority watchlists and targeted actions for marketing teams.
```
