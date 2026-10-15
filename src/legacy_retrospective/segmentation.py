import os
import pandas as pd
import numpy as np

# Configuration
import pathlib
BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()
INPUT_FILE = os.path.join(BASE_DIR, 'data/processed/members_features_with_prob.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data/processed/members_segmented.csv')

def main():
    print("Loading features with churn probabilities...")
    full = pd.read_csv(INPUT_FILE)
    
    print("Applying segmentation rules...")
    # Thresholds based on actual data
    CLV_HIGH    = full['CLV'].quantile(0.75)            # 8940.58
    CLV_MID     = full['CLV'].quantile(0.25)            # 3980.84
    FLIGHTS_MED = full['total_flights_24m'].median()    # 34.0
    
    def assign_segment(row):
        high_value = (row['CLV'] >= CLV_HIGH) and (row['total_flights_24m'] >= FLIGHTS_MED)
        mid_value  = (row['CLV'] >= CLV_MID)  and (row['total_flights_24m'] >= 10)
        churned    = (row['broad_churn'] == 1)
        at_risk    = (row['months_since_last_flight'] >= 3) or churned
        declining  = row['flight_trend'] < -5
        rising     = row['flight_trend'] > 5
        
        if churned:
            return 'Churned / Dormant'
        elif high_value and at_risk:
            return 'At-Risk High Value'
        elif high_value:
            return 'Champions'
        elif mid_value and rising:
            return 'Rising Members'
        elif mid_value and declining:
            return 'Declining Regulars'
        elif mid_value:
            return 'Stable Regulars'
        else:
            return 'Low Engagement'
            
    full['segment'] = full.apply(assign_segment, axis=1)
    
    print("Segment distribution:")
    print(full['segment'].value_counts())
    
    print("Calculating priority score...")
    # Normalizes to 0-83. Higher = more urgent
    full['priority_score'] = full['churn_probability'] * full['CLV'] / 1000
    
    print("Assigning churn risk tiers...")
    full['churn_risk_tier'] = pd.cut(
        full['churn_probability'],
        bins=[-0.1, 0.3, 0.6, 1.1],
        labels=['Low', 'Medium', 'High']
    )
    
    print("Saving segmented data...")
    full.to_csv(OUTPUT_FILE, index=False)
    print(f"Segmentation complete! Output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
