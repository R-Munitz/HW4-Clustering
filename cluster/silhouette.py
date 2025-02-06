import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        #math
        #s(i)= b(i)−a(i)/max(a(i),b(i))
        #a(i) = mean distance from point i to all other points in self cluster - intra cluster distance
        #b(i) = mean distance from point i to all points in nearest (non self) cluster  - nearest cluster distance

        #initialize array of scores
        silhouette_scores = np.zeros(X.shape[0])

        for i in range (X.shape[0]):
            point = X[i]
            assigned_cluster = y[i]

            #get all points in that cluster
            cluster_points = X[y == assigned_cluster]

            # calculate a(i)
            #if cluster has more than one point
            if len(cluster_points) > 1:
                # calculate pair wise distances using cdist
                distances = cdist([point], cluster_points, metric='euclidean')[0] 
                a_i = np.mean(distances[1:]) #skip the first one because it is the distance to itself
            else:
                a_i = 0

        
            #calculate b(i)
            min_b_i = np.inf #initialize to infinity

            #get unique clusters
            unique_clusters = np.unique(y[y != assigned_cluster])
            for cluster in unique_clusters:
                #get all points in cluster
                cluster_points = X[y == cluster]
                if len(cluster_points) > 0:
                    distances = cdist([point], cluster_points, metric='euclidean')[0]
                    mean_distance = np.mean(distances)
                    min_b_i = min(min_b_i, mean_distance) #keeping track of the minimum distance
            
            b_i = min_b_i #minimum distance to nearest cluster

            #calculate silhouette scores
            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0

        return silhouette_scores



            



        


 

