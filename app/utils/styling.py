import streamlit as st

def set_page_config():
    st.set_page_config(
        page_title="Airline Loyalty Analytics",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def inject_custom_css():
    st.markdown("""
        <style>
        .metric-card {
            background-color: #1e1e1e;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            color: #4CAF50;
        }
        .metric-label {
            font-size: 14px;
            color: #888;
            text-transform: uppercase;
        }
        .risk-high {
            background-color: rgba(255, 75, 75, 0.2);
            color: #ff4b4b;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        .risk-medium {
            background-color: rgba(255, 165, 0, 0.2);
            color: orange;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        .risk-low {
            background-color: rgba(76, 175, 80, 0.2);
            color: #4CAF50;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

def render_metric_card(label, value, color_hex="#4CAF50"):
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {color_hex};">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
    """, unsafe_allow_html=True)
