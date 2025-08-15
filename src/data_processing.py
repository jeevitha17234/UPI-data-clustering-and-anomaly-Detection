import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_excel(path)
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
