import pandas as pd
import streamlit as st
import os

@st.cache_data
def load_data():
    """Load the segmented members data for the dashboard"""
    # Look for the file in the processed directory relative to project root
    data_path = os.path.join(os.path.dirname(__file__), '../../data/processed/members_segmented.csv')
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        st.error(f"Data file not found at {data_path}. Please run the data pipelines first.")
        return pd.DataFrame()

@st.cache_data
def load_raw_activity():
    """Load raw flight activity to show time series"""
    # Used for the 24-month sparkline charts
    data_path = os.path.join(os.path.dirname(__file__), '../../Datasets/Customer Flight Activity.csv')
    try:
        # Load and deduplicate to match the processing steps
        fa = pd.read_csv(data_path)
        fa = fa.drop_duplicates(subset=['Loyalty Number', 'Year', 'Month'], keep='first')
        return fa
    except FileNotFoundError:
        return pd.DataFrame()
