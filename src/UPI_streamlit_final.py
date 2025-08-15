import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="UPI Data Dashboard", layout="wide")

# =========================
# Helper Functions
# =========================
@st.cache_data
def load_data(excel_path):
    df = pd.read_excel(excel_path)
    df.columns = df.columns.astype(str).str.strip()
    return df

def clean_data(df):
    cat_cols = ["BankNameSent","BankNameReceived","City","Gender","TransactionType",
                "Status","DeviceType","PaymentMethod","MerchantName","Purpose","PaymentMode","Currency"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower().replace({"nan": np.nan})

    num_cols = ["Amount","RemainingBalance","CustomerAge"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "TransactionDate" in df.columns:
        df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    if "TransactionTime" in df.columns:
        tt = pd.to_datetime(df["TransactionTime"], errors="coerce")
        df["TransactionTime"] = tt.dt.time.where(tt.notna(), None)

    if "TransactionDate" in df.columns:
        if "TransactionTime" in df.columns and df["TransactionTime"].notna().any():
            df["TxnDateTime"] = pd.to_datetime(
                df["TransactionDate"].astype(str).str.strip() + " " +
                df["TransactionTime"].astype(str).str.strip(), errors="coerce")
        else:
            df["TxnDateTime"] = df["TransactionDate"]

    df = df.dropna(subset=["TxnDateTime","Amount"])
    df = df.drop_duplicates(subset=["TransactionID"], keep="last") if "TransactionID" in df.columns else df
    return df

def feature_engineering(df):
    df["hour"] = df["TxnDateTime"].dt.hour
    df["dayofweek"] = df["TxnDateTime"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)
    df["month"] = df["TxnDateTime"].dt.month

    # Rolling stats per customer
    def add_customer_stats(frame):
        f = frame.sort_values(["CustomerAccountNumber","TxnDateTime"]).copy()
        g = f.groupby("CustomerAccountNumber", group_keys=False)
        f["amt_mean_cust"] = g["Amount"].transform(lambda x: x.rolling(10, min_periods=1).mean())
        f["amt_std_cust"]  = g["Amount"].transform(lambda x: x.rolling(10, min_periods=1).std()).fillna(0)
        f["txn_gap_min"] = g["TxnDateTime"].transform(lambda x: x.diff().dt.total_seconds()/60).fillna(1e6)
        f["txn_count_7d"] = g["TxnDateTime"].transform(lambda x: pd.Series(1,index=pd.DatetimeIndex(x)).rolling('7D').sum().values)
        return f

    if {"CustomerAccountNumber","TxnDateTime","Amount"}.issubset(df.columns):
        df = add_customer_stats(df)

    if {"CustomerAccountNumber","MerchantAccountNumber"}.issubset(df.columns):
        pair_counts = df.groupby(["CustomerAccountNumber","MerchantAccountNumber"]).size().rename("cust_merchant_count").reset_index()
        df = df.merge(pair_counts, on=["CustomerAccountNumber","MerchantAccountNumber"], how="left")
    else:
        df["cust_merchant_count"] = np.nan

    df["high_amount_flag"] = (df["Amount"] > df["Amount"].median()).astype(int)
    df["low_balance_flag"] = (df["RemainingBalance"] < df["RemainingBalance"].quantile(0.1)).astype(int) if "RemainingBalance" in df.columns else 0
    return df

def run_clustering(df, features_num, features_cat):
    cluster_features = features_num + features_cat
    pre_cluster = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=False))
            ]), features_num)
        ], remainder="drop"
    )
    Xc = pre_cluster.fit_transform(df[cluster_features])
    best_k, best_score, best_model = None, -1, None
    for k in range(3,7):
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(Xc)
        score = silhouette_score(Xc, labels)
        if score > best_score:
            best_k, best_score, best_model = k, score, km
    df["txn_cluster"] = best_model.fit_predict(Xc)
    return df, best_k, best_score

def run_anomaly_detection(df, features_num, features_cat):
    pre_anom = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=False))
            ]), features_num)
        ], remainder="drop"
    )
    X_anom = pre_anom.fit_transform(df[features_cat + features_num])
    iso = IsolationForest(n_estimators=400, contamination=0.01, random_state=42, n_jobs=-1)
    iso.fit(X_anom)
    df["iso_pred"] = iso.predict(X_anom)
    df["anomaly_score"] = -iso.score_samples(X_anom)
    df["is_suspicious"] = (df["iso_pred"] == -1).astype(int)
    return df

# =========================
# Load & Prepare Data
# =========================
EXCEL_PATH = r"D:\IITB\PLACEMENT\RESUME_PREP\DS PROJ\UPI+Transactions.xlsx"
df = load_data(EXCEL_PATH)
df = clean_data(df)
df = feature_engineering(df)

numerics = ["Amount","hour","dayofweek","is_weekend","amt_mean_cust","amt_std_cust","txn_gap_min","txn_count_7d"]
categoricals = ["DeviceType","PaymentMethod","PaymentMode","City"]

df, best_k, best_score = run_clustering(df, numerics, categoricals)
df = run_anomaly_detection(df, numerics, categoricals)

# =========================
# Sidebar Filters
# =========================
st.sidebar.header("Filters")
min_date, max_date = df["TxnDateTime"].min().date(), df["TxnDateTime"].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
cluster_selected = st.sidebar.multiselect("Select Cluster(s)", options=sorted(df["txn_cluster"].unique()), default=None)
show_suspicious = st.sidebar.checkbox("Show Only Suspicious Transactions", value=False)

filtered_df = df.copy()
filtered_df = filtered_df[(filtered_df["TxnDateTime"].dt.date >= date_range[0]) & 
                          (filtered_df["TxnDateTime"].dt.date <= date_range[1])]
if cluster_selected:
    filtered_df = filtered_df[filtered_df["txn_cluster"].isin(cluster_selected)]
if show_suspicious:
    filtered_df = filtered_df[filtered_df["is_suspicious"]==1]

# =========================
# Dashboard KPIs
# =========================
st.title("💳 UPI Transactions Dashboard")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", len(filtered_df))
col2.metric("Suspicious Transactions", filtered_df["is_suspicious"].sum())
col3.metric("Avg Transaction Amount", f"₹{filtered_df['Amount'].mean():.2f}")
col4.metric("Total Revenue", f"₹{filtered_df['Amount'].sum():.2f}")

# =========================
# Interactive Plots
# =========================
# 1. Daily transactions
daily = filtered_df.set_index("TxnDateTime").resample("D")["TransactionID"].count().reset_index()
fig = px.line(daily, x="TxnDateTime", y="TransactionID", title="Daily Transaction Count")
st.plotly_chart(fig, use_container_width=True)

# 2. Anomaly Heatmap (Hour vs Day)
heat_df = filtered_df.groupby(["dayofweek","hour"])["is_suspicious"].mean().unstack(fill_value=0)
fig = px.imshow(heat_df, labels=dict(x="Hour", y="Day of Week", color="Anomaly Rate"), title="Anomaly Rate Heatmap")
st.plotly_chart(fig, use_container_width=True)

# 3. Amount per Cluster
fig = px.box(filtered_df, x="txn_cluster", y="Amount", color="txn_cluster", title="Transaction Amount by Cluster")
st.plotly_chart(fig, use_container_width=True)

# 4. Top Merchants
top_merchants = filtered_df.groupby("MerchantName")["Amount"].sum().reset_index().sort_values("Amount", ascending=False).head(10)
fig = px.bar(top_merchants, x="MerchantName", y="Amount", title="Top 10 Merchants by Amount")
st.plotly_chart(fig, use_container_width=True)

# =========================
# Suspicious Transactions Table
# =========================
st.subheader("⚠️ Suspicious Transactions")
suspicious_df = filtered_df[filtered_df["is_suspicious"]==1]
st.dataframe(suspicious_df)

# Download buttons
csv = suspicious_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Suspicious Transactions CSV", data=csv, file_name="suspicious_transactions.csv", mime='text/csv')

full_csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Filtered Transactions CSV", data=full_csv, file_name="filtered_transactions.csv", mime='text/csv')
