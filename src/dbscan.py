import numpy as np

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def _find_neighbors(self, X, point_idx):
        """Find all points within `eps` radius of the given point."""
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, labels, point_idx, neighbors, cluster_id):
        """Recursively expand the cluster."""
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if labels[neighbor_idx] == -1:  # Previously marked as noise
                labels[neighbor_idx] = cluster_id
            elif labels[neighbor_idx] == 0:  # Unvisited point
                labels[neighbor_idx] = cluster_id
                new_neighbors = self._find_neighbors(X, neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate([neighbors, new_neighbors])
            i += 1
        return labels

    def fit(self, X):
        n_samples = X.shape[0]
        self.labels_ = np.zeros(n_samples, dtype=int)  # 0 = unvisited, -1 = noise
        cluster_id = 0

        for point_idx in range(n_samples):
            if self.labels_[point_idx] != 0:  # Skip if already visited
                continue

            neighbors = self._find_neighbors(X, point_idx)
            if len(neighbors) < self.min_samples:
                self.labels_[point_idx] = -1  # Mark as noise
            else:
                cluster_id += 1
                self.labels_ = self._expand_cluster(X, self.labels_, point_idx, neighbors, cluster_id)

        return self