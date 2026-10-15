import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import load_data
from utils.styling import set_page_config, inject_custom_css, render_metric_card

set_page_config()
inject_custom_css()

st.title("🧩 2017 Activity Segments")
st.markdown("Segments use activity known by 2017-12-31; observed outcomes are from 2018.")

df = load_data()
if df.empty:
    st.stop()

segments = sorted(df['segment'].dropna().unique())

selected_segment = st.selectbox("Select Segment to Explore", segments)

# Filter data
seg_df = df[df['segment'] == selected_segment]

# Segment logic descriptions
seg_desc = {
    'High Activity, Currently Inactive': "Top-quartile 2017 flight volume and at least 3 months without a flight at the cutoff.",
    'Growing Activity': "Active near the cutoff, with at least 5 more flights in H2 than H1 2017.",
    'Declining Activity': "Active near the cutoff, with at least 5 fewer flights in H2 than H1 2017.",
    'Stable Active': "Active near the cutoff without a large H1-to-H2 change.",
    'No 2017 Activity': "No recorded flights during the 2017 feature window.",
    'Low Activity / Inactive': "Remaining members with low volume or recent inactivity."
}

st.info(f"**Definition:** {seg_desc.get(selected_segment)}")

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    render_metric_card("Size", f"{len(seg_df):,}")
with col2:
    render_metric_card("Avg Flights (2017)", f"{seg_df['total_flights_2017'].mean():.1f}")
with col3:
    render_metric_card("Avg Months Inactive at Cutoff", f"{seg_df['months_since_last_flight'].mean():.1f}")
with col4:
    churn_rate = seg_df['broad_churn'].mean() * 100
    render_metric_card("Observed 2018 Churn", f"{churn_rate:.1f}%", color_hex="#FF4B4B" if churn_rate > 20 else "#4CAF50")

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("2017 Flight Volume")
    fig_flights = px.histogram(seg_df, x='total_flights_2017', nbins=30, marginal="box", color_discrete_sequence=['#2196F3'])
    st.plotly_chart(fig_flights, use_container_width=True)

with col2:
    st.subheader("2017 H2 vs H1 Flight Change")
    fig_trend = px.histogram(seg_df, x='flight_trend_h2_vs_h1_2017', nbins=30, color_discrete_sequence=['#FF9800'])
    st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")
st.subheader("Observed Outcome Breakdown")
outcome_counts = seg_df['broad_churn'].map({0: 'No broad churn in 2018', 1: 'Broad churn in 2018'}).value_counts().reset_index()
outcome_counts.columns = ['2018 outcome', 'Members']
fig_outcome = px.pie(outcome_counts, values='Members', names='2018 outcome', hole=0.5,
                     color='2018 outcome', color_discrete_map={'Broad churn in 2018': '#F44336', 'No broad churn in 2018': '#4CAF50'})
st.plotly_chart(fig_outcome, use_container_width=True)
