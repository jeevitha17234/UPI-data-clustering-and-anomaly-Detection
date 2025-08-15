from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

def run_anomaly_detection(df, features_num=None, features_cat=None):
    if features_num is None:
        features_num = ["Amount","hour","dayofweek","is_weekend","amt_mean_cust","amt_std_cust","txn_gap_min","txn_count_7d"]
    if features_cat is None:
        features_cat = ["DeviceType","PaymentMethod","PaymentMode","City"]

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
