import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
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
df_selected = df[features]

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

st.sidebar.header("⚙️ Clustering Options")
algorithm = st.sidebar.selectbox(
    "Choose Clustering Algorithm", ["K-Means", "DBSCAN", "Hierarchical"]
)

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

st.write("### 📈 Clustering Performance Metrics")
metrics = compute_metrics(df_scaled, df["Cluster"])
st.write(metrics)

if st.button("Run All Algorithms"):
    results = {}
    
    # K-means
    kmeans = KMeans(n_clusters=4)
    results['K-means'] = kmeans.fit_predict(X)
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.5)
    results['DBSCAN'] = dbscan.fit_predict(X)
    
    # Hierarchical
    agg = AgglomerativeClustering(n_clusters=4)
    results['Hierarchical'] = agg.fit_predict(X)
    
    # Create comparison DataFrame
    comparison_df = df[['country']].copy()
    for algo_name, clusters in results.items():
        comparison_df[algo_name] = clusters
    
    st.dataframe(comparison_df)
    
    metrics = []
    for algo_name, clusters in results.items():
        metrics.append({
            'Algorithm': algo_name,
            'Silhouette': silhouette_score(X, clusters),
            'Davies-Bouldin': davies_bouldin_score(X, clusters)
        })
    
    st.table(pd.DataFrame(metrics))

st.sidebar.markdown("### 📥 Download Results")
csv = df.to_csv(index=False).encode()
st.sidebar.download_button("Download Clustered Data", csv, "clustered_data.csv", "text/csv")

st.success("✅ Clustering Completed Successfully!")
