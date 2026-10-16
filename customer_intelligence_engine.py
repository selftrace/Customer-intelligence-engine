import 
import zipfile
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import kruskal
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# we load and clean data
zip_path = "/content/online+retail.zip"
with zipfile.ZipFile(zip_path) as archive:
    name = next(n for n in archive.namelist() if n.lower().endswith(('.xlsx', '.xls')))
    df = pd.read_excel(archive.read(name))

df.columns = df.columns.str.strip()
df = df.dropna(subset=["CustomerID", "InvoiceDate"])
df["InvoiceNo"] = df["InvoiceNo"].astype(str)
df = df[
    ~df["InvoiceNo"].str.startswith("C")
    & (df["Quantity"] > 0)
    & (df["UnitPrice"] > 0)
].copy()
df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Sales"] = df["Quantity"] * df["UnitPrice"]

last_date = df["InvoiceDate"].max()
print(f"Loaded {len(df):,} items | {df['CustomerID'].nunique():,} customers | Data up to: {last_date:%Y-%m-%d}\n")

# RFM Analysis & K-Means Clustering
rfm = df.groupby("CustomerID").agg(
    LastPurchase=("InvoiceDate", "max"),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("Sales", "sum"),
)
rfm["Recency"] = (last_date.normalize() + pd.Timedelta(days=1) - rfm["LastPurchase"]).dt.days
rfm = rfm[["Recency", "Frequency", "Monetary"]].reset_index()

# we normalize and cluster
features = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
scaled = StandardScaler().fit_transform(features)
rfm["Segment"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(scaled).astype(str)

#dimensionality reduction
coords = PCA(n_components=2).fit_transform(scaled)
rfm["PC1"], rfm["PC2"] = coords[:, 0], coords[:, 1]

#visualizing segments
fig_segments = px.scatter(
    rfm, x="PC1", y="PC2", color="Segment",
    hover_data=["CustomerID", "Recency", "Frequency", "Monetary"],
    title="PCA View of Customer Segments"
)
fig_segments.show()

#profile segments
profile = rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].median().round(1)
print("\n--- Customer Segments Profile (Medians) ---")
display(profile)

#cohort retention
orders = df[["CustomerID", "InvoiceNo", "InvoiceDate"]].drop_duplicates()
orders["OrderMonth"] = orders["InvoiceDate"].dt.to_period("M")
first_month = orders.groupby("CustomerID")["OrderMonth"].min().rename("Cohort")
orders = orders.join(first_month, on="CustomerID")
orders["CohortAge"] = (
    (orders["OrderMonth"].dt.year - orders["Cohort"].dt.year) * 12
    + orders["OrderMonth"].dt.month - orders["Cohort"].dt.month
)

cohort_counts = orders.groupby(["Cohort", "CohortAge"])["CustomerID"].nunique().unstack()
retention = cohort_counts.div(cohort_counts.iloc[:, 0], axis=0) * 100
retention.index = retention.index.astype(str)

fig_retention = px.imshow(
    retention, labels={"x": "Months Since First Purchase", "y": "Cohort Month", "color": "Retained (%)"},
    color_continuous_scale="Blues", aspect="auto", zmin=0, zmax=100,
    title="Monthly Customer Retention Heatmap (%)"
)
fig_retention.show()

#statistical testing
groups = [g["Monetary"].values for _, g in rfm.groupby("Segment") if len(g) > 1]
if len(groups) >= 2:
    stat, p_val = kruskal(*groups)
    print(f"\nKruskal-Wallis H-test p-value: {p_val:.4e}")
    print("There is a statistically significant difference in spending habits across customer segments." if p_val < 0.05 else "No significant difference found.")

#churn modeling (90-day horizon)
cutoff = last_date - pd.Timedelta(days=90)
past = df[df["InvoiceDate"] < cutoff].copy()
future_ids = set(df.loc[df["InvoiceDate"] >= cutoff, "CustomerID"])

if not past.empty:
    order_sales = past.groupby(["CustomerID", "InvoiceNo"])["Sales"].sum()
    churn = past.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Sales", "sum"),
    )
    churn["Recency"] = (cutoff.normalize() - churn["LastPurchase"].dt.normalize()).dt.days
    churn["AvgOrderValue"] = order_sales.groupby("CustomerID").mean()
    churn["Churned"] = (~churn.index.isin(future_ids)).astype(int)

    X = churn[["Recency", "Frequency", "Monetary", "AvgOrderValue"]].fillna(0)
    y = churn["Churned"]

    if y.nunique() >= 2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        print(f"\nHoldout Churn Prediction ROC-AUC: {roc_auc_score(y_test, probs):.3f}")

        scored = churn.loc[X_test.index, ["Recency", "Frequency", "Monetary"]].copy()
        scored["Actual_Churn"] = y_test
        scored["Churn_Risk"] = probs
        print("\n--- Top 10 Customers Most Likely to Churn ---")
        display(scored.sort_values("Churn_Risk", ascending=False).head(10))

!pip install streamlit
