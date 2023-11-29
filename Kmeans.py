import numpy as np
from Cluster import cluster
from sklearn.datasets import make_blobs


class Kmeans(cluster):
    def __init__(self, num_clusters, max_iter):
        super(Kmeans, self).__init__()

        self.num_clusters = num_clusters
        self.max_iter = max_iter
    
    def init_centroids(self, data, k, random_state = 12):
        #randomly places inital clusters, returns numpy array of k centroids
        #data = numpy array of data
        #k = num clusters
        
        np.random.seed(random_state)
        numRows, numCols = data.shape
        centroids = np.empty(shape=(k, numCols))

        for i in range(k):
            point = np.random.randint(0, numRows - 1)
            centroids[i] = data[point]

        return centroids

    def assignClusters(self, data, centroids):
        numRows, numCols = data.shape
        num_clusters = len(centroids)
        centroidAssignments = [0] * numRows

        for row in range(numRows):
            mindist = np.inf
            for i in range(num_clusters):
                dist_sum = 0
                for col in range(numCols):
                    dist_sum += (centroids[i, col] - data[row, col]) ** 2
                #euclidian distance = sqrt(x^2+y^2+z^2+...)
                dist_sum = dist_sum**0.5
                if dist_sum < mindist:
                    mindist = dist_sum
                    centroidAssignments[row] = i

        return centroidAssignments


    def moveCentroids(self, data, centroidAssignments):
        # centroidAssignments = each row's nearest centroid
        # find average location of all points associated with each centroid
        # move centroids to average "central" locations
        numCols = data.shape[1]
        num_clusters = len(np.unique(centroidAssignments))
        centroids = np.empty(shape=(num_clusters, numCols))

        for clusternum in range(num_clusters):
            numPoints = 0
            newPosition = np.zeros((1, numCols))

            for x in range(len(centroidAssignments)):
                if centroidAssignments[x] == clusternum:
                    numPoints += 1
                    newPosition += data[x]

            centroids[clusternum] = newPosition / numPoints

        return centroids
    
    def fit(self,data):
        #first initialize clusters
        centroids = self.init_centroids(data, self.num_clusters)

        #loop though the maximum number of times or if we find constant clusters
        for i in range(self.max_iter):
            centroidAssignments = self.assignClusters(data, centroids)
            
            new_centroids = self.moveCentroids(data, centroidAssignments)

            if (np.all(centroids == new_centroids)):
                break
            else:
                centroids = new_centroids
            
            #If our centroids didn't move, stop, if they did, run again
        return centroids, centroidAssignments


    if __name__ == "__main__":
        data = make_blobs(n_samples=1000, n_features=5)
        
        kmeans = Kmeans(num_clusters = 5, max_iter = 1000)

        final_centroids, assignments = kmeans.fit(data)
        