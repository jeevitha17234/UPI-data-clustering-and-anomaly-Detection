from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_clustering(df, features_num=None, features_cat=None):
    if features_num is None:
        features_num = ["Amount","hour","dayofweek","is_weekend","amt_mean_cust","amt_std_cust","txn_gap_min","txn_count_7d"]
    if features_cat is None:
        features_cat = ["DeviceType","PaymentMethod","PaymentMode","City"]

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
