# Write your k-means unit tests here
from cluster import (
        KMeans, 
        Silhouette, 
        make_clusters,
        plot_clusters,
        plot_multipanel)
import pytest
import numpy as np

def test_kmeans_init():

    #test edge cases

    #test value error is raised if k is not an integer
    with pytest.raises(ValueError):
        km = KMeans(k=1.5)

    #test value error is raised if k is 0
    with pytest.raises(ValueError):
        km = KMeans(k=0)

    #test value error is raised if tol is not a float
    with pytest.raises(ValueError):
        km = KMeans(k=1, tol=1)

    #test value error is raised if tol is 0
    with pytest.raises(ValueError):
        km = KMeans(k=1, tol=0)
    
    #test value error is raised when max_iter is not an integer
    with pytest.raises(ValueError):
        km = KMeans(k=1, max_iter=1.5)
    
    #test value error is raised when max_iter is 0
    with pytest.raises(ValueError):
        km = KMeans(k=1, max_iter=0)

def test_kmeans_fit():

    #test edge cases

    #test that mat is not empty
    with pytest.raises(ValueError):
        km = KMeans(k=1)
        km.fit(np.array([]))
    
    #test that mat is a 2D matrix
    with pytest.raises(ValueError):
        km = KMeans(k=1)
        km.fit(np.array([1, 2, 3]))

    #test that value error is raised if k is > number of data points
    with pytest.raises(ValueError):
        km = KMeans(k=100)
        km.fit(np.random.rand(10, 10))

    #test on simple data clustering
    data = np.array([
        [1, 2], [1, 3], [2, 2], [2, 3],  # cluster 1
        [8, 8], [9, 9], [8, 9], [9, 8]   # cluster 2
    ])
    
    km = KMeans(k=2)
    km.fit(data)

    #expected centers are ~mean of data points
    expected_centers = np.array([[1.5, 2.5], [8.5, 8.5]])

def test_kmeans_predict():

    #check if expected centers match assigned centers

    #sort ( to match correct centers)
    sorted_clusters = np.sort(km.cluster_centers, axis = 0)
    sorted_predicted_clusters = np.sort(expected_centers, axis = 0)

    #check if they are closely matched 
    assert (np.allclose(sorted_clusters, sorted_predicted_clusters, atol = 0.5))

    #test if data points are correctly assigned with data containing two distinct clusters
    data = np.array([[0, 0], [1, 1], [9, 9], [10, 10]])
    km = KMeans(k=2)
    km.fit(data)
    predictions = km.predict(data)
    assert len(set(predictions)) == 2




