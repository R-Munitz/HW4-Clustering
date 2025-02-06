import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        #error handling
        #k must be an integer >0
        if not isinstance(k, int):
            raise ValueError("k must be an integer")
        if k <= 0:
            raise ValueError("k must be greater than 0")
        
        #tol must be a float >0
        if not isinstance(tol, float):
            raise ValueError("tol must be a float")
        if tol <= 0:
            raise ValueError("tol must be greater than 0")
        
        #max_iter must be an integer >0
        if not isinstance(max_iter, int):
            raise ValueError("max_iter must be an integer")
        if max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")
        
        self.k = k # number of clusters desired
        self.tol = tol # tolerance for stopping criterion (when to stop iterating)
        self.max_iter = max_iter # maximum number of iterations to run before quitting 
        #check iter within a reasonable range?
        self.cluster_centers = None
        self.dataset = None
        self.cluster_assignments = None


    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        #pseudocode
        #randomly intialize k clusters
        #iterate until convergence or max_iterations
            #for each data point in the matrix:
                #calc distance to each cluster center
                #assign data point to closest cluster 
            #for each cluster:
                #calculate cluster center - mean of all points
            #check to see if it converged
                #if difference is less than tolerance, stop
                #else, update new cluster center

        #error handling
        
        #mat must not be an empty matrix
        if mat.size == 0:
            raise ValueError("mat must not be an empty matrix")
        #mat must be a 2D numpy array
        if not isinstance(mat, np.ndarray):
            raise ValueError("mat must be a 2D numpy array")
        if mat.ndim != 2:
            raise ValueError("mat must be a 2D numpy array")
     
        #k must be < the number of data points - move to fit()
        if self.k >= mat.shape[0]:
            raise ValueError("k must be less than the number of data points")
        
        self.dataset = mat

        #set seed for reproducibility
        np.random.seed(45)

        #randomly pick k data points from mat as initial cluster centers
        k_cluster_indices = np.random.choice(mat.shape[0], self.k, replace=False) #don't allow duplicates
        self.cluster_centers = mat[k_cluster_indices]

        #until convergence or max_iterations 
        for _ in range(self.max_iter):

            #for every data point in matrix, calculate distances to cluster centers
            distances = cdist(mat, self.cluster_centers, metric='euclidean')

            #assign each data point to the nearest cluster, and store the cluster assignments 
            self.cluster_assignments = np.argmin(distances, axis=1)

            #cluster_assignments = np.array([np.argmin([self.calculate_Euclidean_distance(datapoint, center)     
                                                    #for center in self.cluster_centers])
                                                       # for datapoint in mat])
            
            #calculate new cluster centroid based on mean of all points assigned to cluster
            updated_cluster_centers = np.array([ mat[self.cluster_assignments == i].mean(axis=0) if np.any(self.cluster_assignments == i)  
                                                else self.cluster_centers[i]  # keep previous cluster center
                                                for i in range(self.k)
            ])
            
            #check for convergence (is difference between new clusters and current clusters smaller than the tolerance?)
            if np.all(np.abs(updated_cluster_centers - self.cluster_centers) < self.tol):
                break
            else: #update centers
                self.cluster_centers = updated_cluster_centers
        
        
    '''
    def calculate_Euclidean_distance(self, data_point: np.ndarray, cluster_center: np.ndarray) -> float: #rename to private func? Use scipy cdist instead
        """
        Calculates the Euclidean distance between a data point and a cluster center.

        inputs:
            data_point: np.ndarray
                A 1D array representing a data point
            cluster_center: np.ndarray
                A 1D array representing a cluster center
        """
        #calculate the Euclidean distance between a data point and a cluster center
        distance = np.sqrt(np.sum((data_point - cluster_center)**2)) #euclidean distance formula
        return distance
    '''    

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        #assigning cluster labels to each data point based on distance to cluster centers

        #psuedocode
        #check for errors ( empty matrix, uninitialized cluster centers, mismatched features, non-numeric values etc) 
       
        #for each data point in the matrix:
            #calculate distance to each cluster centroid
            #find closest cluster
            #store predicted cluster label
        #return 1D array of cluster labels


        if self.cluster_centers is None:
            raise ValueError("Model was not fitted yet!")

        #predict labels for new data points
        #cluster_labels = np.array([np.argmin([self.calculate_Euclidean_distance(datapoint, center)     
                                                    #for center in self.cluster_centers])
                                                        #for datapoint in mat])
        #calculate distances 
        distances = cdist(mat, self.cluster_centers, metric='euclidean')

        #assign label to data point
        cluster_labels = np.argmin(distances, axis=1)
        
        return cluster_labels


    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        if self.cluster_centers is None or self.cluster_assignments is None:
            raise ValueError("Model was not fitted yet! Call the fit method first.")
        
        #calculate error (sqrd Euclidean distance between each datapoint and assigned cluster, sum, divide by nr of points = mean squared error)
        """
        total_squared_error = sum(
              self.calculate_Euclidean_distance(datapoint, self.cluster_centers[self.cluster_assignments[i]]) ** 2
                                        for i, datapoint in enumerate(self.dataset)
        )
        """
        #won't work, use cdist instead of method no longer instantiated, need to calc distance between datapoints and assigned cluster?
        distances = cdist(self.dataset, self.cluster_centers, metric='euclidean') #for each data point in mat, calculate distance to cluster centers

        # get the cluster assignments - not stored, so calculate min distance to cluster center
        #cluster_assignments = np.argmin(distances, axis=1)

        #get minimum distances (distance to assigned cluster)
        min_distances = distances [np.arange(self.dataset.shape[0]), self.cluster_assignments]

        #calculate squared mean error
        squared_mean_error = np.mean(min_distances ** 2)


        #mean_squared_error = total_squared_error / len(self.dataset)

        return squared_mean_error
                                    
       
    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self.cluster_centers