# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 23:06:08 2020

@author: olamide
"""

#importing the libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#Loading dataset 
dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:,3:5].values

#KMEANS CLUSTERING


#finding the optimal clusters using the elbow method 
from sklearn.cluster import KMeans
wcss =[]
for i in range (1,11):
    kmeans =KMeans(n_clusters = i, init ="k-means++", n_init = 10,max_iter = 300, random_state = None)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss) 
plt.show()

#The Elbow method shows that the maximum number of cluster is 5
kmeans =KMeans(n_clusters = 5 , init ="k-means++", n_init = 10,max_iter = 300, random_state = None)
pred = kmeans.fit_predict(x)

#Visualizing the result
plt.scatter(x[pred==0,0], x[pred==0,1], c ='blue', s=100, label ='Sensible')
plt.scatter(x[pred==1,0], x[pred==1,1], c ='green', s=100, label ='Target')
plt.scatter(x[pred==2,0], x[pred==2,1], c ='black', s=100, label ='Careless')
plt.scatter(x[pred==3,0], x[pred==3,1], c ='red',   s=100, label ='standard')
plt.scatter(x[pred==4,0], x[pred==4,1], c ='cyan',  s=100, label ='Careful')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label ='Centroid' )
plt.ylabel("spending score")
plt.xlabel("Amount spent")
plt.legend()
plt.show()




#HIERARCHIAL CLUSTERING
import scipy.cluster.hierarchy as sch
#Using Dendrogram to find the optimal number of clusters
dendrogram = sch.dendrogram(sch.linkage(x, method ='ward'))
plt.title("Dendrogram")
plt.xlabel('Customers')
plt.ylabel("Euclidean Distance")
plt.show()



#Fitting Hierarchy clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, linkage = 'ward', affinity = 'euclidean')
y_hc = hc.fit_predict(x)


#Visualizing the result
plt.scatter(x[y_hc==0,0], x[y_hc==0,1], c ='blue', s=100, label ='Sensible')
plt.scatter(x[y_hc==1,0], x[y_hc==1,1], c ='green', s=100, label ='Target')
plt.scatter(x[y_hc==2,0], x[y_hc==2,1], c ='black', s=100, label ='Careless')
plt.scatter(x[y_hc==3,0], x[y_hc==3,1], c ='red', s=100, label ='standard')
plt.scatter(x[y_hc==4,0], x[y_hc==4,1], c ='cyan', s=100, label ='Careful')

plt.ylabel("spending score")
plt.xlabel("Amount spent")
plt.legend()
plt.show()