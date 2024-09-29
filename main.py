@app.route('/run-kmeans', methods=['POST'])
def run_kmeans():
    data = np.array(request.json['data'])
    method = request.json['method']

    if method == "manual":
        centroids = np.array(request.json['centroids'])
        kmeans = KMeans(k=3, init_method="manual")
        kmeans.centroids = centroids  # Set manually chosen centroids
    else:
        kmeans = KMeans(k=3, init_method=method)
    
    labels = kmeans.fit(data)
    centroids = kmeans.centroids.tolist()
    
    return jsonify(labels=labels.tolist(), centroids=centroids)
