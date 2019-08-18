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
 
#Q2    
daily_returns = Stocks30.pct_change().fillna(0)
daily_returns_mean = daily_returns.mean()
daily_returns_mean_reshaped = daily_returns_mean.values.reshape(30,1)
cov_matrix = daily_returns.cov()
weights = np.asarray([0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2])
portfolio_return = round(np.sum(daily_returns_mean_reshaped * weights) * 252,2)
portfolio_std_dev = round(np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(252),2)
print('Annualised Return = ',portfolio_return)    
print('Volatility = ',portfolio_std_dev)


#q3
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

Optimal_cluster = 2 
plt.show()    
plt.scatter(X[:,0],X[:,1], label='True Position')
kmeans = KMeans(n_clusters=Optimal_cluster)
kmeans.fit(X)
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_)
kmeans.labels_

