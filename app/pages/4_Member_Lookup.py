import streamlit as st
import pandas as pd
from utils.data_loader import load_data, load_raw_activity
from utils.charts import get_member_sparkline
from utils.styling import set_page_config, inject_custom_css

set_page_config()
inject_custom_css()

st.title("🔍 Member Lookup")
st.caption("Member profile is a historical snapshot: 2017 activity predicts the separately observed 2018 outcome.")

df = load_data()
fa_raw = load_raw_activity()

if df.empty:
    st.stop()

# Search
search_id = st.text_input("Enter Loyalty Number (e.g. 100018)", "")

if search_id:
    try:
        search_id = int(search_id)
        member = df[df['Loyalty Number'] == search_id]
        
        if member.empty:
            st.error(f"Member {search_id} not found.")
        else:
            member = member.iloc[0]
            
            # Action logic
            actions = {
                'High Activity, Currently Inactive': ['Test a route-relevant reminder plus a capped points offer against usual communication'],
                'Growing Activity': ['Test a progress message without an incentive'],
                'Declining Activity': ['Test a route-relevant reminder against usual communication'],
                'Stable Active': ['Continue usual communication and monitor flight activity'],
                'No 2017 Activity': ['Test a low-cost reactivation message'],
                'Low Activity / Inactive': ['Use a low-cost message; reserve expensive offers for tested cohorts']
            }
            rec_actions = actions.get(member['segment'], ['Monitor'])
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"Profile: {search_id}")
                st.markdown(f"**Segment:** <span class='risk-{member['churn_risk_tier'].lower()}'>{member['segment']}</span>", unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Flights in 2017", f"{member['total_flights_2017']:.0f}")
                c2.metric("2017 Activity Segment", member['segment'])
                c3.metric("Enrollment", f"{member['Enrollment Year']}-{str(member['Enrollment Month']).zfill(2)}")
                
                c4, c5, c6 = st.columns(3)
                c4.metric("Province", member['Province'])
                c5.metric("Months Inactive at Cutoff", member['months_since_last_flight'])
                c6.metric("Observed 2018 Broad Churn", "Yes" if member['broad_churn'] else "No")
                
                st.markdown("---")
                st.subheader("Flight Activity: Feature and Outcome Windows")
                if not fa_raw.empty:
                    m_fa = fa_raw[fa_raw['Loyalty Number'] == search_id]
                    fig_spark = get_member_sparkline(m_fa)
                    st.plotly_chart(fig_spark, use_container_width=True)
                
            with col2:
                # Churn Gauge
                st.subheader("Cross-Validated 2018 Risk Score")
                import plotly.graph_objects as go
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = member['churn_probability'] * 100,
                    number = {'suffix': "%"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Estimated risk"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 60], 'color': "orange"},
                            {'range': [60, 100], 'color': "salmon"}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(t=30, b=0, l=10, r=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

                st.caption("Historical cross-validated model score; it is not a live or causal estimate.")
                
                st.markdown("### Recommended Actions")
                for i, act in enumerate(rec_actions, 1):
                    st.markdown(f"{i}. {act}")
                    
    except ValueError:
        st.error("Please enter a valid numeric Loyalty Number.")
