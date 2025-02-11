def sqrt(n, precision=1e-10):
    """Manually calculates the square root using Newton’s method."""
    if n == 0:
        return 0
    guess = n / 2.0
    while True:
        new_guess = (guess + n / guess) / 2
        if abs(new_guess - guess) < precision:
            return new_guess
        guess = new_guess

def calculate_distance(x1, y1, x2, y2):
    """Calculates Euclidean distance without using math.sqrt()."""
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def assign_clusters(datapoints, centroids):
    """Assigns each datapoint to the nearest centroid."""
    clusters = {i: [] for i in range(len(centroids))}
    
    for x, y in datapoints:
        closest = min(range(len(centroids)), key=lambda i: calculate_distance(x, y, *centroids[i]))
        clusters[closest].append([x, y])
    
    return clusters

def calculate_new_centroids(clusters, old_centroids):
    """Computes new centroids as the mean of assigned points."""
    new_centroids = []
    
    for cluster in clusters.values():
        if cluster:  # Avoid empty clusters
            x_avg = sum(p[0] for p in cluster) / len(cluster)
            y_avg = sum(p[1] for p in cluster) / len(cluster)
            new_centroids.append([x_avg, y_avg])
        else:
            new_centroids.append(old_centroids[len(new_centroids)])  
    
    return new_centroids

def KMeans(datapoints, k=3, max_iterations=100):
    """Performs K-Means clustering."""
    centroids = datapoints[:k]  # Initialize centroids using first k points
    
    for _ in range(max_iterations):
        clusters = assign_clusters(datapoints, centroids)
        new_centroids = calculate_new_centroids(clusters, centroids)
        
        if new_centroids == centroids:
            break  # Stop if centroids don’t change
        
        centroids = new_centroids 

    return centroids, clusters

# **New dataset (changed values)**
datapoints = [
    [1, 8], [2, 6], [9, 3], [4, 5], 
    [7, 2], [6, 6], [3, 1], [5, 9]
]

final_centroids, final_clusters = KMeans(datapoints)

# Output results
print("Final Centroids:", final_centroids)
for cluster_id, points in final_clusters.items():
    print(f"Cluster {cluster_id + 1}: {points}")
