import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scripts.clustering import perform_clustering
from scripts.visualization import plot_clusters, plot_dendrogram
from scripts.evaluation import compute_metrics

st.set_page_config(page_title="Country Clustering", layout="wide")
st.title("🌍 Country Clustering Using ML Algorithms")

st.sidebar.header("🔼 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("data/Country-data.csv")  

st.write("### 📊 Data Preview")
st.write(df.head())

features = st.multiselect("Select Features for Clustering", df.columns[1:], default=df.columns[1:3])
if not features:
    st.error("⚠️ Please select at least two features!")
    st.stop()

df_selected = df[features]
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

# --- CLUSTERING OPTIONS ---
st.sidebar.header("⚙️ Clustering Options")
algorithm = st.sidebar.selectbox("Choose Clustering Algorithm", ["K-Means", "DBSCAN", "Hierarchical"])

params = {}
if algorithm == "K-Means":
    params["n_clusters"] = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
elif algorithm == "DBSCAN":
    params["eps"] = st.sidebar.slider("Epsilon (eps)", 0.1, 2.0, 0.5)
    params["min_samples"] = st.sidebar.slider("Min Samples", 2, 10, 5)
elif algorithm == "Hierarchical":
    params["linkage"] = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average", "single"])

df["Cluster"] = perform_clustering(df_scaled, algorithm, params)

st.write("### 🎨 Cluster Visualization")
fig1 = plot_clusters(df, features)
st.pyplot(fig1)

if algorithm == "Hierarchical":
    st.write("### 🌳 Hierarchical Clustering Dendrogram")
    fig2 = plot_dendrogram(df_scaled)
    st.pyplot(fig2)

st.write("### 🌍 Clustered Countries")

for cluster_id, cluster_df in df.groupby("Cluster"):
    with st.expander(f" View Cluster {cluster_id} ({len(cluster_df)} countries)"):
        st.table(cluster_df[["country"]])


st.write("### 📈 Clustering Performance Metrics")
metrics = compute_metrics(df_scaled, df["Cluster"])
st.write(metrics)

st.sidebar.markdown("### 📥 Download Results")
csv = df.to_csv(index=False).encode()
st.sidebar.download_button("Download Clustered Data", csv, "clustered_data.csv", "text/csv")

st.success("✅ Clustering Completed Successfully!")
