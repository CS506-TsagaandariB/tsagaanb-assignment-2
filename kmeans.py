import numpy as np
import random

class KMeans:
    def __init__(self, k=3, init_method="random"):
        self.k = k
        self.init_method = init_method
        self.centroids = None

    def initialize_centroids(self, data):
        if self.init_method == "random":
            self.centroids = data[np.random.choice(data.shape[0], self.k, replace=False)]
        elif self.init_method == "farthest_first":
            self.centroids = self._farthest_first(data)
        elif self.init_method == "kmeans++":
            self.centroids = self._kmeans_plus_plus(data)

    def _farthest_first(self, data):
        centroids = [data[random.randint(0, len(data)-1)]]
        for _ in range(1, self.k):
            distances = np.array([min(np.linalg.norm(x-c) for c in centroids) for x in data])
            next_centroid = data[np.argmax(distances)]
            centroids.append(next_centroid)
        return np.array(centroids)

    def _kmeans_plus_plus(self, data):
        centroids = [data[random.randint(0, len(data)-1)]]
        for _ in range(1, self.k):
            # Compute the squared distance of each point to the nearest centroid
            distances = np.array([min(np.linalg.norm(x - c)**2 for c in centroids) for x in data])
            probabilities = distances / distances.sum()
            cumulative_probs = np.cumsum(probabilities)
            random_val = random.uniform(0, 1)
            next_centroid_idx = np.where(cumulative_probs >= random_val)[0][0]
            centroids.append(data[next_centroid_idx])
        return np.array(centroids)

    def assign_clusters(self, data):
        distances = np.array([[np.linalg.norm(point - centroid) for centroid in self.centroids] for point in data])
        return np.argmin(distances, axis=1)

    def update_centroids(self, data, labels):
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(self.k)])
        return new_centroids

    def fit(self, data):
        self.initialize_centroids(data)
        for _ in range(100):  # max 100 iterations
            labels = self.assign_clusters(data)
            new_centroids = self.update_centroids(data, labels)
            if np.all(new_centroids == self.centroids):
                break
            self.centroids = new_centroids
        return labels
