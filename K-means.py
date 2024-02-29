# importing packages
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 

# reading data
cust_df = pd.read_csv("Cust_Segmentation.csv")
cust_df.head()


# pre-processing
# we donot need Address here(base of our data)
df = cust_df.drop('Address', axis=1)
df.head()

# normilizing
from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet

# modeling
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(Clus_dataSet)
labels = k_means.labels_
print(labels)

