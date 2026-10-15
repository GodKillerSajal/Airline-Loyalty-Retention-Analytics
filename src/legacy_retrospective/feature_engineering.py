import os
import pandas as pd
import numpy as np
from itertools import product

# Configuration
import pathlib
BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(BASE_DIR, 'Datasets/')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data/processed/')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("Starting feature engineering...")
    
    # 1. Load Data
    print("Loading data...")
    lh = pd.read_csv(os.path.join(DATA_DIR, 'Customer Loyalty History.csv'))
    fa = pd.read_csv(os.path.join(DATA_DIR, 'Customer Flight Activity.csv'))
    
    # Step 0: Deduplicate and Build Full 24-Month Grid
    print("Deduplicating flight activity...")
    fa_dedup = fa.drop_duplicates(subset=['Loyalty Number', 'Year', 'Month'], keep='first').copy()
    
    print("Building full 24-month grid for all members...")
    all_members = lh['Loyalty Number'].unique()
    grid = pd.DataFrame(
        [(m, y, mo) for m, y, mo in product(all_members, [2017, 2018], range(1, 13))],
        columns=['Loyalty Number', 'Year', 'Month']
    )
    
    fa_full = grid.merge(
        fa_dedup[['Loyalty Number','Year','Month','Total Flights','Distance',
                  'Points Accumulated','Points Redeemed','Dollar Cost Points Redeemed']],
        on=['Loyalty Number','Year','Month'], how='left'
    ).fillna(0)
    
    fa_full = fa_full.sort_values(['Loyalty Number','Year','Month']).reset_index(drop=True)
    
    # Step 1: Build Member-Level Aggregate Features
    print("Building member-level aggregate features...")
    # Recency
    fa_active = fa_full[fa_full['Total Flights'] > 0].copy()
    fa_active['period'] = fa_active['Year'] * 100 + fa_active['Month']
    last_active = (fa_active.groupby('Loyalty Number')['period']
                   .max().reset_index()
                   .rename(columns={'period': 'last_active_period'}))
    
    def months_since(period, obs_period=201812):
        y, m = divmod(int(period), 100)
        oy, om = divmod(obs_period, 100)
        return (oy - y) * 12 + (om - m)
        
    last_active['months_since_last_flight'] = last_active['last_active_period'].apply(months_since)
    
    # Year-split flights
    flights_by_year = (
        fa_full.groupby(['Loyalty Number','Year'])['Total Flights']
        .sum()
        .unstack(fill_value=0)
        .rename(columns={2017: 'flights_2017', 2018: 'flights_2018'})
        .reset_index()
    )
    
    # Seasonal flights
    fa_full['quarter'] = pd.cut(fa_full['Month'], bins=[0,3,6,9,12], labels=['Q1','Q2','Q3','Q4'])
    quarterly = (
        fa_full.groupby(['Loyalty Number','quarter'], observed=False)['Total Flights']
        .sum()
        .unstack(fill_value=0)
        .rename(columns={'Q1':'q1_flights','Q2':'q2_flights','Q3':'q3_flights','Q4':'q4_flights'})
        .reset_index()
    )
    
    # Core Aggregation
    member_features = fa_full.groupby('Loyalty Number').agg(
        total_flights_24m      = ('Total Flights', 'sum'),
        avg_monthly_flights    = ('Total Flights', 'mean'),
        max_monthly_flights    = ('Total Flights', 'max'),
        active_months          = ('Total Flights', lambda x: (x > 0).sum()),
        std_monthly_flights    = ('Total Flights', 'std'),
        total_distance         = ('Distance', 'sum'),
        avg_distance_per_month = ('Distance', 'mean'),
        total_points_acc       = ('Points Accumulated', 'sum'),
        total_points_red       = ('Points Redeemed', 'sum'),
        total_dollar_redeemed  = ('Dollar Cost Points Redeemed', 'sum'),
        months_with_redemption = ('Points Redeemed', lambda x: (x > 0).sum()),
    ).reset_index()
    
    member_features = (member_features
        .merge(flights_by_year, on='Loyalty Number', how='left')
        .merge(quarterly,       on='Loyalty Number', how='left')
    )
    
    # Derived Features
    member_features['activity_rate'] = member_features['active_months'] / 24
    member_features['flight_trend']  = member_features['flights_2018'] - member_features['flights_2017']
    member_features['redemption_rate'] = np.where(
        member_features['total_points_acc'] > 0,
        member_features['total_points_red'] / member_features['total_points_acc'], 0
    )
    member_features['has_redeemed'] = (member_features['total_points_red'] > 0).astype(int)
    member_features['avg_dist_per_flight'] = np.where(
        member_features['total_flights_24m'] > 0,
        member_features['total_distance'] / member_features['total_flights_24m'], 0
    )
    member_features['seasonal_concentration'] = (
        member_features[['q1_flights','q2_flights','q3_flights','q4_flights']].max(axis=1) /
        (member_features['total_flights_24m'] + 0.001)
    )
    
    # Step 2: Build max_consec_zeros Feature (Churn Signal)
    print("Computing max consecutive zeros...")
    def max_consec_zeros(group):
        flew = (group['Total Flights'] > 0).values
        max_z = cur_z = 0
        for f in flew:
            if not f:
                cur_z += 1
                max_z = max(max_z, cur_z)
            else:
                cur_z = 0
        return max_z
        
    consec = (
        fa_full
        .sort_values(['Loyalty Number','Year','Month'])
        .groupby('Loyalty Number', group_keys=False)
        .apply(lambda g: pd.Series({'max_consec_zeros': max_consec_zeros(g)}), include_groups=False)
        .reset_index()
    )
    
    member_features = member_features.merge(consec, on='Loyalty Number', how='left')
    
    # Step 3: Merge Demographics + Recency
    print("Merging demographics and recency...")
    full = member_features.merge(
        last_active[['Loyalty Number','months_since_last_flight','last_active_period']],
        on='Loyalty Number', how='left'
    )
    full['months_since_last_flight'] = full['months_since_last_flight'].fillna(24)
    
    full = full.merge(lh, on='Loyalty Number', how='left')
    
    full['tenure_months'] = (
        (2018 - full['Enrollment Year']) * 12 + (12 - full['Enrollment Month'])
    )
    
    # Salary cleaning + imputation
    full['Salary'] = full['Salary'].where(full['Salary'] >= 0, np.nan)
    original_null = lh.set_index('Loyalty Number')['Salary'].isna()
    original_negative = lh.set_index('Loyalty Number')['Salary'].lt(0).fillna(False)
    
    sal_by_group   = full.groupby(['Province','Education','Loyalty Card'], observed=False)['Salary'].transform('median')
    sal_by_prov    = full.groupby('Province', observed=False)['Salary'].transform('median')
    full['Salary'] = full['Salary'].fillna(sal_by_group).fillna(sal_by_prov).fillna(full['Salary'].median())
    
    full['salary_imputed'] = full['Loyalty Number'].map(
        lambda lid: int(original_null.get(lid, False) or original_negative.get(lid, False))
    )
    
    # Ordinal encodings
    card_map = {'Star': 1, 'Nova': 2, 'Aurora': 3}
    full['card_tier_num'] = full['Loyalty Card'].map(card_map)
    
    edu_map = {'High School or Below': 1, 'College': 2, 'Bachelor': 3, 'Master': 4, 'Doctor': 5}
    full['edu_num'] = full['Education'].map(edu_map)
    
    full['is_promo_member'] = (full['Enrollment Type'] == '2018 Promotion').astype(int)
    
    # Step 4: Build Churn Labels
    print("Building churn labels...")
    full['formally_churned'] = full['Cancellation Year'].notna().astype(int)

    # Silent churn: 6+ consecutive zero-flight months, NOT formally cancelled.
    # IMPORTANT: Exclude 2018 Promotion members from this rule.
    # Promo members enrolled in 2018, so their 2017 months are filled with
    # structural zeros (they hadn't joined yet). This produces max_consec_zeros >= 13
    # for ALL promo members regardless of actual engagement — not a churn signal.
    full['silent_churned']   = (
        (full['formally_churned'] == 0) &
        (full['max_consec_zeros'] >= 6) &
        (full['is_promo_member'] == 0)        # exclude promo members
    ).astype(int)
    full['broad_churn'] = (
        (full['formally_churned'] == 1) | (full['silent_churned'] == 1)
    ).astype(int)
    # For promo members: only formally cancelled ones are churned
    # (silent-churn label is unreliable due to structural zeros)
    promo_formally_churned = (
        (full['is_promo_member'] == 1) & (full['formally_churned'] == 1)
    ).sum()
    print(f"  Note: {full['is_promo_member'].sum()} promo members excluded from "
          f"silent-churn rule. Only their {promo_formally_churned} formal cancellations count.")
    
    print("Churn breakdown:")
    print(full[['formally_churned','silent_churned','broad_churn']].sum())
    
    # Save processed features
    output_path = os.path.join(OUTPUT_DIR, 'members_features.csv')
    print(f"Saving features to {output_path}...")
    full.to_csv(output_path, index=False)
    print("Feature engineering complete!")

if __name__ == "__main__":
    main()
