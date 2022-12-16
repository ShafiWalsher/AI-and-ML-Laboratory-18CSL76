import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pandas as pd

data=pd.read_csv("pgm7.csv")
df1=pd.DataFrame(data)
f1 = df1['attribute1'].values
f2 = df1['attribute2'].values
X=np.asarray(list(zip(f1,f2))) #X holds the input samples
#Applying EM algorithm
gmm = GaussianMixture (n_components = 4).fit(X)
lables = gmm.predict(X)
#Plot of predicted labels for each data sample
plt.plot()
colors = ['b', 'g', 'r','c']
markers = ['o', 'P', 's','*']
for i, l in enumerate(lables):
    plt.plot(f1[i], f2[i], color=colors[l], marker=markers[l])
    plt.xlim([-3, 3])
    plt.ylim([0, 10])
plt.show()
#Applying KMeans clustering algorithm
kmeans_model = KMeans(n_clusters=4).fit(X)
plt.plot()
colors = ['b', 'g', 'r','c']
markers = ['o', 'P', 's','*']
plt.plot()
for i, l in enumerate(kmeans_model.labels_):
    plt.plot(f1[i], f2[i], color=colors[l], marker=markers[l])
    plt.xlim([-3, 3])
    plt.ylim([0, 10])
plt.show()