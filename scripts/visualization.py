import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch

def plot_clusters(df, features):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[features[0]], y=df[features[1]], hue=df["Cluster"], palette="viridis")
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title("Clusters Visualization")
    return plt

def plot_dendrogram(data):
    plt.figure(figsize=(10, 6))
    sch.dendrogram(sch.linkage(data, method="ward"))
    plt.title("Hierarchical Clustering Dendrogram")
    return plt
