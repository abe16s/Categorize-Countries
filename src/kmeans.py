import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None

    def _initialize_centroids(self, X):
        # Randomly pick initial centroids from data points
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def _assign_clusters(self, X):
        # Assign each point to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        # Compute new centroids as the mean of assigned points
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centroids

    def fit(self, X):
        self.centroids = self._initialize_centroids(X)
        for _ in range(self.max_iter):
            old_centroids = self.centroids.copy()
            self.labels = self._assign_clusters(X)
            self.centroids = self._update_centroids(X, self.labels)
            # Check for convergence
            if np.linalg.norm(self.centroids - old_centroids) < self.tol:
                break

    def predict(self, X):
        return self._assign_clusters(X)