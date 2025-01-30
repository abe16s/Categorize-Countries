import numpy as np
from scipy.spatial.distance import pdist, squareform

class AgglomerativeClusteringWard:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.labels_ = None

    def fit(self, X):
        n_samples = X.shape[0]
        self.labels_ = np.arange(n_samples)
        clusters = [[i] for i in range(n_samples)]
        cluster_sizes = np.ones(n_samples, dtype=int)
        pairwise_dists = squareform(pdist(X, metric='sqeuclidean'))

        while len(clusters) > self.n_clusters:
            # Find closest clusters (i, j)
            min_dist = np.inf
            min_i, min_j = -1, -1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = pairwise_dists[i, j]
                    if dist < min_dist:
                        min_dist = dist
                        min_i, min_j = i, j

            # Capture cluster sizes and indices before deletion
            n_i = cluster_sizes[min_i]
            n_j = cluster_sizes[min_j]
            new_size = n_i + n_j

            # Precompute distances for the new cluster
            new_dists = []
            for k in range(len(clusters)):
                if k != min_i and k != min_j:
                    d_ik = pairwise_dists[min_i, k]
                    d_jk = pairwise_dists[min_j, k]
                    d_ij = pairwise_dists[min_i, min_j]
                    n_k = cluster_sizes[k]
                    d_new = ((n_i + n_k) * d_ik + (n_j + n_k) * d_jk - n_k * d_ij) / (n_i + n_j + n_k)
                    new_dists.append(d_new)

            # Merge clusters and update structures
            new_cluster = clusters[min_i] + clusters[min_j]
            del clusters[max(min_i, min_j)]
            del clusters[min(min_i, min_j)]
            clusters.append(new_cluster)

            # Update cluster sizes
            cluster_sizes = np.delete(cluster_sizes, [min_i, min_j])
            cluster_sizes = np.append(cluster_sizes, new_size)

            # Update distance matrix
            pairwise_dists = np.delete(pairwise_dists, [min_i, min_j], axis=0)
            pairwise_dists = np.delete(pairwise_dists, [min_i, min_j], axis=1)
            pairwise_dists = np.pad(pairwise_dists, ((0, 1), (0, 1)), mode='constant', constant_values=0)
            pairwise_dists[-1, :-1] = new_dists
            pairwise_dists[:-1, -1] = new_dists

        # Assign final labels
        self.labels_ = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for point in cluster:
                self.labels_[point] = cluster_id

        return self