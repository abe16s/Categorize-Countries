import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_clusters(X, labels, centroids, title):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    if centroids:
        centroids_pca = pca.transform(centroids)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
    if centroids:
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', s=200, marker='X')
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()