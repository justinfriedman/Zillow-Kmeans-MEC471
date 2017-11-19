import numpy as np
from sklearn.cluster import KMeans
import pandas
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
# https://www.slideshare.net/SarahGuido/kmeans-clustering-with-scikitlearn
#help from that link
data = pandas.read_stata('data/ames_train_no_string.dta', convert_missing=True)
data = data.values

kmeans = KMeans()
kmeans.fit(data)

k_range = range(1,14)
k_means_var = [KMeans(n_clusters=k).fit(data) for k in k_range]

centroids = [X.cluster_centers_ for X in k_means_var]
k_euclid = [cdist(data, cent, 'euclidean') for cent in centroids]

dist = [np.min(ke, axis=1) for ke in k_euclid]

wcss = [sum(d**2) for d in dist]

tss = sum(pdist(data)**2)/data.shape[0]

bss = tss - wcss
