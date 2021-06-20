#!/usr/bin/env python
# coding: utf-8

# In[251]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn import metrics


# In[252]:


dataset = pd.read_csv('E:/TSF/Iris.csv')


# In[253]:


dataset.shape


# In[254]:


dataset.describe()


# In[255]:


dataset.info()


# In[256]:


dataset['Species'].unique()


# In[257]:


a = pd.DataFrame(dataset)
new_data = a.drop(columns = ['Species', 'Id'])
new_data.info()


# In[258]:


new_data.shape


# In[259]:


within_cluster_sum_of_square = [] 

cluster_range = range(1,15)
for k in cluster_range:
    km = KMeans(n_clusters=k)
    km = km.fit(new_data)
    within_cluster_sum_of_square.append(km.inertia_)


# In[260]:


plt.plot(cluster_range, within_cluster_sum_of_square,'go--', color = 'red')
plt.grid()
plt.xlabel('cluster_range')
plt.ylabel('Within-Cluster-sum-of-square')
plt.title('The Elbow Method')
plt.show()


# In[261]:



model = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
predictions = model.fit_predict(new_data)


# In[262]:


x = new_data.iloc[:, [0, 1, 2, 3]].values
plt.scatter(x[predictions == 0, 0], x[predictions == 0, 1], s=25, c= 'red', label = 'iris-setosa')
plt.scatter(x[predictions == 1, 0], x[predictions == 1, 1], s=25, c= 'green', label = 'iris-versicolour')
plt.scatter(x[predictions == 2, 0], x[predictions == 2, 1], s=25, c= 'blue', label = 'iris-virginica')


plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:,1], s=80, color = '#ffff00', label = 'Centroids')
plt.legend(bbox_to_anchor=(1.36,1.34))
plt.grid() 
plt.show()

