
# coding: utf-8

# # Module 6 - Clustering for Diverse portfolio analysis

#    ### Welcome to the Answer notebook for Module 6 ! 
# Make sure that you've submitted the module 5 notebook and unlocked Module 6 yourself before you start coding here
# 

# #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ### Query 6.1 
# Create a table/dataframe with the closing prices of 30 different stocks, with 10 from each of the caps

# In[ ]:


import numpy as np 
import pandas as pd
import warnings
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
stocks = list(os.listdir())
stocks.pop(31)
stocks.pop(29)
stocks.pop(27)

cols = []
for i in stocks:
    cols.append(i[:-4])
Stocks30 = pd.DataFrame(columns = cols)
for i in range(len(stocks)):
    data1 = pd.read_csv(stocks[i])
    Stocks30[cols[i]] = data1['Close Price']


# ### Query 6.2
# Calculate average annual percentage return and volatility of all 30 stocks over a theoretical one year period

# In[ ]:


daily_returns = Stocks30.pct_change().fillna(0)
daily_returns_mean = daily_returns.mean()
daily_returns_mean_reshaped = daily_returns_mean.values.reshape(30,1)
cov_matrix = daily_returns.cov()
weights = np.asarray([0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2])
portfolio_return = round(np.sum(daily_returns_mean_reshaped * weights) * 252,2)
portfolio_std_dev = round(np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(252),2)
print('Annualised Return = ',portfolio_return)    
print('Volatility = ',portfolio_std_dev)


# ### Query 6.3
# Cluster the 30 stocks according to their mean annual Volatilities and Returns using K-means clustering. Identify the optimum number of clusters using the Elbow curve method

# In[ ]:


mean_returns30 = []
mean_std30 = []
daily_returns = Stocks30.pct_change().fillna(0)
daily_std = daily_returns.std()
for i in range(len(daily_returns_mean_reshaped)):
    mean_returns30.append(round(((daily_returns_mean_reshaped[i] * weights[i]) * 252).mean(),2))
    mean_std30.append(round(daily_std[i]*np.sqrt(252),2))
    
X = np.array([mean_returns30,mean_std30]).transpose()

mms = MinMaxScaler()
mms.fit(X)
X = mms.transform(X)

Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')

Optimal_cluster = 3 
plt.show()    
plt.scatter(X[:,0],X[:,1], label='True Position')
kmeans = KMeans(n_clusters=Optimal_cluster)
kmeans.fit(X)
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_)
print(kmeans.labels_)


# ### Query 6.4
# Prepare a separate Data frame to show which stocks belong to the same cluster 

# In[ ]:


clustered_df = pd.DataFrame(columns = cols1)
for i in range(kmeans.n_clusters):
    q = list(np.where(kmeans.labels_ == i)[0])
    for j in q:
        print("In cluster "+str(i)+" : ", cols[j])

