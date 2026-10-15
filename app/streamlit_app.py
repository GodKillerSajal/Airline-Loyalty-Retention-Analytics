import streamlit as st
from utils.styling import set_page_config, inject_custom_css

set_page_config()
inject_custom_css()

st.title("✈️ Airline Loyalty Retention Analytics")

st.markdown("""
Explore a historical airline loyalty analysis and a proposed retention experiment:

- **1_Overview_Dashboard**: Leakage-aware evaluation of 2018 churn using 2017-only features.
- **2_Churn_Watchlist**: Cross-validated historical scores for inspecting missed and captured cases.
- **3_Segment_Explorer**: Activity segments defined using the 2017 feature window.
- **4_Member_Lookup**: Historical member activity, risk score, and 2018 outcome.
- **5_Retention_Playbook**: Proposed intervention and randomized test plan.

### Project Context
The model cutoff is **2017-12-31** and its outcome window is **2018**. No campaign assignment or treatment outcomes were supplied, so the project does not estimate a retention offer's causal impact. The dashboard is a historical case study, not a live airline system.
""")

st.info("👈 Please select a tool from the sidebar to begin.")
