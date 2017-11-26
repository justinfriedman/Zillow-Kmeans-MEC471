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

# plt.scatter(data[:, 0], data[:, 1], c=k_range, s=50, cmap='viridis')
plt.scatter(data[:, 0], data[:, 1], c='black', s=200, alpha=0.5)


for i in k_range:
    # select only data observations with cluster label == i
    ds = data[np.where(centroids==i)]
    # plot the data observations
    plt.plot(ds[:,0],ds[:,1],'o')
    # plot the centroids
    lines = plt.plot(data[i,0],data[i,1],'kx')
    # make the centroid x's bigger
    plt.setp(lines,ms=15.0)
    plt.setp(lines,mew=2.0)
plt.show()
