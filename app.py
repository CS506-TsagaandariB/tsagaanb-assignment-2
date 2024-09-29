from flask import Flask, render_template, jsonify, request
import numpy as np
import random

app = Flask(__name__)

# Generate a random dataset
def generate_dataset(n_points=200):
    data = np.random.randn(n_points, 2) * 5
    return data.tolist()

# Initialize centroids
def initialize_centroids(data, k, method="random", manual_centroids=None):
    if method == "random":
        centroids = random.sample(data, k)
    elif method == "farthest_first":
        centroids = [random.choice(data)]
        while len(centroids) < k:
            distances = [min([np.linalg.norm(np.array(p) - np.array(c)) for c in centroids]) for p in data]
            centroids.append(data[np.argmax(distances)])
    elif method == "kmeans++":
        centroids = [random.choice(data)]
        for _ in range(1, k):
            distances = [min([np.linalg.norm(np.array(p) - np.array(c)) for c in centroids]) for p in data]
            probs = distances / np.sum(distances)
            cumulative_probs = np.cumsum(probs)
            r = random.random()
            next_centroid = data[np.searchsorted(cumulative_probs, r)]
            centroids.append(next_centroid)
    elif method == "manual" and manual_centroids:
        centroids = manual_centroids
    return centroids

# KMeans algorithm
def kmeans(data, centroids):
    clusters = [[] for _ in centroids]
    for point in data:
        distances = [np.linalg.norm(np.array(point) - np.array(c)) for c in centroids]
        cluster = np.argmin(distances)
        clusters[cluster].append(point)

    new_centroids = [np.mean(cluster, axis=0).tolist() if len(cluster) > 0 else c for cluster, c in zip(clusters, centroids)]
    return new_centroids, clusters

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_dataset', methods=['GET'])
def generate_dataset():
    # Increase the number of points, e.g., 500 points
    num_points = 500  
    data = np.random.uniform(low=-10, high=10, size=(num_points, 2)).tolist()
    return jsonify({'data': data})
    
@app.route('/initialize_centroids', methods=['POST'])
def initialize():
    data = request.json['data']
    k = int(request.json['k'])
    method = request.json['method']
    manual_centroids = request.json.get('manual_centroids', None)
    centroids = initialize_centroids(data, k, method, manual_centroids)
    return jsonify({'centroids': centroids})

@app.route('/step_kmeans', methods=['POST'])
def step_kmeans():
    data = request.json['data']
    centroids = request.json['centroids']
    new_centroids, clusters = kmeans(data, centroids)
    return jsonify({'new_centroids': new_centroids, 'clusters': clusters})

@app.route('/run_to_convergence', methods=['POST'])
def run_to_convergence():
    data = request.json['data']
    centroids = request.json['centroids']
    converged = False
    while not converged:
        new_centroids, _ = kmeans(data, centroids)
        if np.array_equal(new_centroids, centroids):
            converged = True
        centroids = new_centroids
    return jsonify({'centroids': centroids})

if __name__ == '__main__':
    app.run(debug=True)
