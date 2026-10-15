import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def get_donut_chart(df, names_col, values_col=None, title=None, hole=0.6, colors=None):
    if values_col:
        counts = df.groupby(names_col)[values_col].sum().reset_index()
    else:
        counts = df[names_col].value_counts().reset_index()
        counts.columns = [names_col, 'count']
        values_col = 'count'
        
    fig = px.pie(
        counts, 
        values=values_col, 
        names=names_col, 
        hole=hole, 
        title=title,
        color=names_col if colors else None,
        color_discrete_map=colors
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False, margin=dict(t=40, b=0, l=0, r=0))
    return fig

def get_histogram(df, x_col, title, color_col=None, nbins=30):
    fig = px.histogram(df, x=x_col, color=color_col, nbins=nbins, title=title,
                       marginal="box", opacity=0.7, barmode='overlay')
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    return fig

def get_activity_heatmap(fa_df):
    """Generates a heatmap of flight activity across months"""
    monthly = fa_df.groupby(['Year', 'Month'])['Total Flights'].sum().reset_index()
    monthly['Month_Name'] = pd.to_datetime(monthly['Month'], format='%m').dt.month_abbr
    
    # Pivot for heatmap
    pivot = monthly.pivot(index='Year', columns='Month', values='Total Flights').fillna(0)
    # Ensure all 12 months exist
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = 0
            
    pivot = pivot[sorted(pivot.columns)]
    
    fig = px.imshow(
        pivot.values,
        labels=dict(x="Month", y="Year", color="Total Flights"),
        x=[pd.to_datetime(str(m), format='%m').month_abbr for m in pivot.columns],
        y=pivot.index,
        aspect="auto",
        color_continuous_scale="Blues"
    )
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10))
    return fig

def get_member_sparkline(member_fa):
    """Bar chart for 24 months of a single member's activity"""
    # Create complete grid
    years = [2017, 2018]
    months = list(range(1, 13))
    grid = pd.DataFrame([(y, m) for y in years for m in months], columns=['Year', 'Month'])
    
    merged = grid.merge(member_fa[['Year', 'Month', 'Total Flights']], on=['Year', 'Month'], how='left').fillna(0)
    merged['Period'] = merged['Year'].astype(str) + "-" + merged['Month'].astype(str).str.zfill(2)
    
    fig = px.bar(merged, x='Period', y='Total Flights')
    fig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0), 
        xaxis_title=None, 
        yaxis_title=None,
        height=200
    )
    return fig
