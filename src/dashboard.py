import streamlit as st
from src.data_processing import load_data, clean_data, feature_engineering
from src.clustering import run_clustering
from src.anomaly_detection import run_anomaly_detection
from src.visualizations import plot_daily_txns, plot_anomaly_heatmap

EXCEL_PATH = "data/sample_UPI_transactions.xlsx"
df = load_data(EXCEL_PATH)
df = clean_data(df)
df = feature_engineering(df)
df, best_k, best_score = run_clustering(df)
df = run_anomaly_detection(df)

st.title("💳 UPI Transactions Dashboard")

# Sidebar filters
st.sidebar.header("Filters")
date_range = st.sidebar.date_input("Date Range", [df["TxnDateTime"].min().date(), df["TxnDateTime"].max().date()])
clusters = st.sidebar.multiselect("Select Cluster(s)", sorted(df["txn_cluster"].unique()), default=None)
show_suspicious = st.sidebar.checkbox("Show Only Suspicious Transactions", value=False)

filtered_df = df[(df["TxnDateTime"].dt.date >= date_range[0]) & (df["TxnDateTime"].dt.date <= date_range[1])]
if clusters: filtered_df = filtered_df[filtered_df["txn_cluster"].isin(clusters)]
if show_suspicious: filtered_df = filtered_df[filtered_df["is_suspicious"]==1]

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", len(filtered_df))
col2.metric("Suspicious Transactions", filtered_df["is_suspicious"].sum())
col3.metric("Avg Transaction Amount", f"₹{filtered_df['Amount'].mean():.2f}")
col4.metric("Total Revenue", f"₹{filtered_df['Amount'].sum():.2f}")

# Plots
st.plotly_chart(plot_daily_txns(filtered_df), use_container_width=True)
st.plotly_chart(plot_anomaly_heatmap(filtered_df), use_container_width=True)

# Suspicious Transactions Table
st.subheader("⚠️ Suspicious Transactions")
st.dataframe(filtered_df[filtered_df["is_suspicious"]==1])
