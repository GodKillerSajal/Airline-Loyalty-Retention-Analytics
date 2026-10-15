import streamlit as st
import pandas as pd
from utils.data_loader import load_data, load_raw_activity
from utils.charts import get_donut_chart, get_histogram, get_activity_heatmap
from utils.styling import set_page_config, inject_custom_css, render_metric_card

set_page_config()
inject_custom_css()

st.title("📊 2018 Retention Risk Evaluation")
st.caption("Historical holdout study: predictors use activity through 2017-12-31; outcomes are measured during 2018.")

# Load data
df = load_data()
fa_raw = load_raw_activity()

if df.empty:
    st.stop()

# Top KPIs
total_members = len(df)
formal_churn = df['formally_churned'].sum()
broad_churn = df['broad_churn'].sum()
top_decile = df.nlargest(max(1, int(len(df) * 0.10)), 'churn_probability')
top_decile_recall = top_decile['broad_churn'].sum() / max(1, broad_churn)

col1, col2, col3, col4 = st.columns(4)
with col1:
    render_metric_card("Total Members", f"{total_members:,}")
with col2:
    render_metric_card("Formal Churn Rate", f"{(formal_churn/total_members)*100:.1f}%", color_hex="#FFA500")
with col3:
    render_metric_card("Broad Churn Rate", f"{(broad_churn/total_members)*100:.1f}%", color_hex="#FF4B4B")
with col4:
    render_metric_card("2018 Churn Captured in Top Score Decile", f"{top_decile_recall*100:.1f}%", color_hex="#FF4B4B")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Segment Distribution")
    # Custom colors for segments
    segment_colors = {
        'Stable Active': '#4CAF50',
        'Growing Activity': '#8BC34A',
        'Declining Activity': '#FFC107',
        'High Activity, Currently Inactive': '#FF9800',
        'No 2017 Activity': '#F44336',
        'Low Activity / Inactive': '#9E9E9E'
    }
    fig_seg = get_donut_chart(df, names_col='segment', title=None, colors=segment_colors)
    st.plotly_chart(fig_seg, use_container_width=True)

with col2:
    st.subheader("Cross-Validated Risk Scores")
    fig_hist = get_histogram(df, 'churn_probability', title=None, color_col='broad_churn')
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Monthly Flight Activity (2017-2018)")
    if not fa_raw.empty:
        fig_heat = get_activity_heatmap(fa_raw)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Raw flight activity data unavailable.")

with col2:
    st.subheader("Observed 2018 Broad Churn by 2017 Risk Tier")
    tier_churn = df.groupby('churn_risk_tier')['broad_churn'].agg(['mean', 'size']).reset_index()
    tier_churn['Churn Rate %'] = tier_churn['mean'] * 100
    
    import plotly.express as px
    fig_bar = px.bar(tier_churn, x='churn_risk_tier', y='Churn Rate %', color='churn_risk_tier',
                     color_discrete_map={'High': '#FF4B4B', 'Medium': '#FFA500', 'Low': '#4CAF50'},
                     hover_data=['size'])
    fig_bar.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_bar, use_container_width=True)
