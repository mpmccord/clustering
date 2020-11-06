#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Clustering of Iris and Heart Disease Dataset
"""
Author: Mel McCord 
Date: 11/2/2020
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from cluster_functions import *


# In[2]:


# PCA Analysis of Heart Failure
heart, heart_names = arr_csv("heart_failure_clinical_records_dataset.csv")
norm_heart = (heart - np.mean(heart, axis=0)) / np.std(heart, axis=0)
cov_matrix = np.cov(norm_heart, rowvar=False)
plt.imshow(cov_matrix)
plt.colorbar()
plt.show()
# removing the classification column
print(heart.shape)
X, P, e_scaled = pca_svd(heart)
scatter3D(P, 0, 1, 2)
total_retained = 0
print()
print(heart_names)
print("Maximum visualized clustering retained: ")

X_rec = reconstruct(heart, 3)
x = heart[:, 0]
y = heart[:, 1]
x = X_rec[:, 0]
y = X_rec[:, 1]
fig, ax = plt.subplots(2, 1)
plt.subplots_adjust(hspace=0.8)
scatter3D(heart, 2, 1, 0)
scatter3D(X_rec, 2, 1, 0)


# In[3]:


# Performing PCA on Iris
iris, iris_names = arr_csv("iris.data")
iris_rec = reconstruct(iris, iris.shape[1])
X, P, e_scaled = pca_svd(iris)
print()
print(iris_names)
print("Maximum visualized clustering retained: ")
fig, ax = plt.subplots(2, 1)
plt.subplots_adjust(hspace=0.8)
ax[0].plot(iris[:, 0], iris[:, 1], 'ob', alpha=0.3, c='r')
ax[1].plot(iris_rec[:, 0], iris_rec[:, 1], 'or', alpha=0.3, c='b')
ax[0].set_title("Original Iris Dataset")
ax[1].set_title("Reconstructed Iris Dataset")
ax[1].set_xlabel(iris_names[0])
ax[1].set_ylabel(iris_names[1])
ax[0].set_xlabel(iris_names[0])
ax[0].set_ylabel(iris_names[1])
plt.show()


# In[4]:


# Performing clustering on iris
# cluster_analysis("iris.data", k=3, class_col=4)


# In[ ]:


# Performing clustering on heart failure dataset
cluster_analysis("heart_failure_clinical_records_dataset.csv", k=2, class_col=12)

