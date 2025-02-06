# Write your k-means unit tests here
import KMeans
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

    #test 


