import pandas as pd
import streamlit as st
from pathlib import Path

from utils.styling import inject_custom_css, set_page_config

set_page_config()
inject_custom_css()

st.title("🧪 Retention Experiment Plan")
st.markdown(
    "The source data contains no campaign assignment or treatment outcomes. "
    "This page proposes a randomized test; it does not report a measured lift."
)

playbook_data = [
    {
        "Historical segment": "High Activity, Currently Inactive",
        "Eligibility trigger": "At least 3 months since the last flight at the scoring date",
        "Treatment hypothesis": "Personalized route reminder plus a capped points offer",
        "Control": "Business-as-usual communication",
        "Primary outcome": "Any flight within 90 days of assignment",
        "Guardrails": "Incremental incentive cost, net contribution, opt-outs / complaints",
    },
    {
        "Historical segment": "Declining Activity",
        "Eligibility trigger": "H2 2017 flights at least 5 below H1 2017",
        "Treatment hypothesis": "Route-relevant reminder without a discount",
        "Control": "Business-as-usual communication",
        "Primary outcome": "Any flight within 90 days of assignment",
        "Guardrails": "Incremental incentive cost, net contribution, opt-outs / complaints",
    },
    {
        "Historical segment": "No 2017 Activity",
        "Eligibility trigger": "No recorded flights in the 2017 feature window",
        "Treatment hypothesis": "Low-cost reactivation message",
        "Control": "Business-as-usual communication",
        "Primary outcome": "Any flight within 90 days of assignment",
        "Guardrails": "Contact cost, net contribution, opt-outs / complaints",
    },
]

st.subheader("Proposed protocol")
for item in playbook_data:
    with st.expander(f"{item['Historical segment']} — {item['Eligibility trigger']}"):
        st.write(f"**Treatment:** {item['Treatment hypothesis']}")
        st.write(f"**Control:** {item['Control']}")
        st.write(f"**Primary outcome:** {item['Primary outcome']}")
        st.write(f"**Guardrails:** {item['Guardrails']}")

st.markdown(
    "**Design:** randomize by loyalty number, allocate 1:1, stratify by predicted risk tier, "
    "analyze intent-to-treat, and measure one pre-registered 90-day outcome. "
    "The historical 2018 return rate is only a planning benchmark; it is not the control arm."
)

try:
    project_root = Path(__file__).resolve().parents[2]
    power = pd.read_csv(project_root / "results" / "retention_ab_power.csv")
    st.subheader("Illustrative sample-size requirements")
    st.caption("Two-sided α=0.05, 80% power, equal allocation, with a 10% unusable-follow-up allowance.")
    st.dataframe(power, use_container_width=True, hide_index=True)
except FileNotFoundError:
    st.info("Run `python src/temporal_retention_analysis.py` to generate the historical baseline and power table.")

st.warning(
    "No treatment effect, uplift, or profit impact can be calculated from these files because no randomized "
    "campaign assignment, offer cost, or margin outcome was supplied."
)

st.download_button(
    label="Download Proposed Test Plan",
    data=pd.DataFrame(playbook_data).to_csv(index=False).encode("utf-8"),
    file_name="retention_experiment_plan.csv",
    mime="text/csv",
)
