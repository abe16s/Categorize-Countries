import numpy as np
from scipy.spatial.distance import pdist, squareform

class AgglomerativeClustering:
    def __init__(self, n_clusters=2, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
        self.dendrogram = None

    def _compute_initial_distances(self, X):
        # Compute pairwise distances
        return squareform(pdist(X, metric='euclidean'))

    def _single_linkage(self, dist_matrix, cluster_ids):
        # Find the minimum distance between clusters
        min_dist = np.inf
        merge_a, merge_b = -1, -1

        # Iterate over all pairs of clusters
        for i in range(len(cluster_ids)):
            for j in range(i+1, len(cluster_ids)):
                # Get all points in clusters i and j
                points_i = cluster_ids[i]
                points_j = cluster_ids[j]
                # Compute minimum distance between points in the two clusters
                current_min = np.min(dist_matrix[np.ix_(points_i, points_j)])
                if current_min < min_dist:
                    min_dist = current_min
                    merge_a, merge_b = i, j
        return merge_a, merge_b, min_dist

    def fit(self, X):
        n_samples = X.shape[0]
        # Initialize each point as its own cluster
        clusters = [[i] for i in range(n_samples)]
        # Compute initial distance matrix
        dist_matrix = self._compute_initial_distances(X)
        # Track merge history for dendrogram (not fully implemented here)
        self.dendrogram = []

        # Merge clusters until we reach n_clusters
        while len(clusters) > self.n_clusters:
            # Find clusters to merge (using single linkage)
            merge_a, merge_b, distance = self._single_linkage(dist_matrix, clusters)
            # Merge cluster b into cluster a
            clusters[merge_a].extend(clusters[merge_b])
            del clusters[merge_b]
            # Update distance matrix (this is simplified and inefficient!)
            # Note: A full implementation would recompute distances for the new cluster
            # For simplicity, we'll just track merges here
            self.dendrogram.append((merge_a, merge_b, distance, len(clusters[merge_a])))

        # Assign labels
        self.labels_ = np.zeros(n_samples, dtype=int)
        for idx, cluster in enumerate(clusters):
            for point in cluster:
                self.labels_[point] = idx

        return self