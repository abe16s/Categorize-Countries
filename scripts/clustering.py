from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

def perform_clustering(data, algorithm, params):
    if algorithm == "K-Means":
        model = KMeans(n_clusters=params["n_clusters"], random_state=42)
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
    elif algorithm == "Hierarchical":
        model = AgglomerativeClustering(linkage=params["linkage"])
    return model.fit_predict(data)
