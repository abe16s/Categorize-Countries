from sklearn.metrics import silhouette_score, davies_bouldin_score

def compute_metrics(data, labels):
    metrics = {}
    if len(set(labels)) > 1:
        metrics["Silhouette Score"] = silhouette_score(data, labels)
        metrics["Davies-Bouldin Index"] = davies_bouldin_score(data, labels)
    else:
        metrics["Silhouette Score"] = "Not Applicable (1 Cluster Detected)"
        metrics["Davies-Bouldin Index"] = "Not Applicable (1 Cluster Detected)"
    return metrics
