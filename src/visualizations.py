import plotly.express as px

def plot_daily_txns(df):
    daily = df.set_index("TxnDateTime").resample("D")["TransactionID"].count().reset_index()
    fig = px.line(daily, x="TxnDateTime", y="TransactionID", title="Daily Transaction Count")
    return fig

def plot_anomaly_heatmap(df):
    heat_df = df.groupby(["dayofweek","hour"])["is_suspicious"].mean().unstack(fill_value=0)
    fig = px.imshow(heat_df, labels=dict(x="Hour", y="Day of Week", color="Anomaly Rate"), title="Anomaly Rate Heatmap")
    return fig
