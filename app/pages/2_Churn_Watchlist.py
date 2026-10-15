import streamlit as st
import pandas as pd
from utils.data_loader import load_data
from utils.styling import set_page_config, inject_custom_css

set_page_config()
inject_custom_css()

st.title("🚨 Historical Risk Review")
st.markdown("Cross-validated 2017 risk scores paired with the observed 2018 outcome. Actions are hypotheses; no campaign effect was measured.")

df = load_data()
if df.empty:
    st.stop()

# Filters
st.sidebar.header("Filters")
prov_filter = st.sidebar.multiselect("Province", options=df['Province'].unique())
risk_filter = st.sidebar.multiselect("Risk Tier", options=['High', 'Medium', 'Low'], default=['High', 'Medium'])

# Apply filters
filtered_df = df.copy()
if prov_filter:
    filtered_df = filtered_df[filtered_df['Province'].isin(prov_filter)]
if risk_filter:
    filtered_df = filtered_df[filtered_df['churn_risk_tier'].isin(risk_filter)]

# Sort by priority score
filtered_df = filtered_df.sort_values('churn_probability', ascending=False)

st.write(f"Showing **{len(filtered_df)}** members matching criteria.")

# Action mapping logic; these actions are proposed, not experimentally validated.
def get_action(segment):
    actions = {
        'High Activity, Currently Inactive': 'Test a personalized route reminder with a capped points offer',
        'Growing Activity': 'Test a progress message without an incentive',
        'Declining Activity': 'Test a route-relevant reminder against usual communication',
        'Stable Active': 'Continue usual communication; monitor activity',
        'No 2017 Activity': 'Test a low-cost reactivation message',
        'Low Activity / Inactive': 'Use a low-cost message; avoid an expensive offer until tested'
    }
    return actions.get(segment, 'Monitor')

# Prepare display table
display_df = filtered_df.head(50)[
    ['Loyalty Number', 'segment', 'churn_probability', 'total_flights_2017', 'months_since_last_flight', 'broad_churn']
].copy()

display_df['churn_probability'] = (display_df['churn_probability'] * 100).map("{:.1f}%".format)
display_df['Recommended Action'] = display_df['segment'].apply(get_action)

# Rename columns for display
display_df = display_df.rename(columns={
    'segment': 'Segment',
    'churn_probability': 'Estimated 2018 Risk Score',
    'total_flights_2017': 'Flights in 2017',
    'months_since_last_flight': 'Months Inactive at 2017 Cutoff',
    'broad_churn': 'Observed 2018 Broad Churn'
})

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

st.download_button(
    label="Download Watchlist as CSV",
    data=display_df.to_csv(index=False).encode('utf-8'),
    file_name='historical_risk_review.csv',
    mime='text/csv'
)
