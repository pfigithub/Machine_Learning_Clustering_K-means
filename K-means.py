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