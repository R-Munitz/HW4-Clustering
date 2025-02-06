import numpy as np
import sklearn as sk
from sklearn.metrics import silhouette_score
from cluster import (
        KMeans, 
        Silhouette, 
        make_clusters,
        plot_clusters,
        plot_multipanel)

# write your silhouette score unit tests here
def test_silhouette_score():

    #compare sklearn silhouette score with my silhouette score

    #create clusters
    clusters, labels = make_clusters(k=4, scale=1)

    #fit kmeans
    km = KMeans(k=4)
    km.fit(clusters)

    #predict
    pred = km.predict(clusters)

    #score using my silhouette score
    my_silhouette_score= Silhouette().score(clusters, pred)
    #score using sklearn silhouette score
    sklearn_silhouette_score = silhouette_score(clusters, pred)

    #compare scores 
    #how to choose tolerance?
    assert np.allclose(my_silhouette_score, sklearn_silhouette_score, atol = 0.1)

    #check edge cases (no scores) are handled (not edge cases about the data itself)









