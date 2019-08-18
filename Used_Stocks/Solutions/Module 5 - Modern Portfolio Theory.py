
# coding: utf-8

# # Module 5 - Modern Portfolio Theory

#    ### Welcome to the Answer notebook for Module 5 ! 
# Make sure that you've submitted the module 4 notebook and unlocked Module 5 yourself before you start coding here
# 

# #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ### Query 5.1 
# 5.1 For your chosen stock, calculate the mean daily return and daily standard deviation of returns, and then just annualise them to get mean expected annual return and volatility of that single stock. **( annual mean = daily mean * 252 , annual stdev = daily stdev * sqrt(252) )**

# In[ ]:


import pandas as pd
import numpy as np
import math


# In[ ]:


dataRCOM = pd.read_csv('RCOM.csv')
dataRCOM['Daily Return'] = (dataRCOM['Close Price']).pct_change() 
dataRCOM['Daily Return'] = dataRCOM['Daily Return'].replace([np.inf, -np.inf], np.nan)
dataRCOM = dataRCOM.dropna()
print("Mean Daily Return :",dataRCOM['Daily Return'].mean())

dataRCOM['Daily Standard Deviation'] = (dataRCOM['Close Price']).pct_change() 
dataRCOM['Daily Standard Deviation'] = dataRCOM['Daily Standard Deviation'].replace([np.inf, -np.inf], np.nan)
dataRCOM = dataRCOM.dropna()
print("Daily Standard Deviation :",dataRCOM['Daily Standard Deviation'].std())

annual_mean =  -0.0002659107842087934 * 252
print("Annual Mean: "+ str(annual_mean))
annual_stdev = 0.013515857199631161 * math.sqrt(252)
print("Annual Standard Deviation: "+ str(annual_stdev))


# ### Query 5.2
# Now, we need to diversify our portfolio. Build your own portfolio by choosing any 5 stocks, preferably of different sectors and different caps. Assume that all 5 have the same weightage, i.e. 20% . Now calculate the annual returns and volatility of the entire portfolio ( Hint : Don't forget to use the covariance )

# In[ ]:


jklakshmi_data = pd.read_csv('JKLAKSHMI.csv')
raymond_data = pd.read_csv('RAYMOND.csv')
wipro_data = pd.read_csv('wipro_stock_data.csv')
itc_data = pd.read_csv('itc_stock_data.csv')
airtel_data = pd.read_csv('airtel_stock_data.csv')

cols = ['JKLAKSHMI','RAYMOND','WIPRO','ITC','AIRTEL']
data = {
        'JKLAKSHMI' : jklakshmi_data['Close Price'].iloc[:492],
        'RAYMOND' : raymond_data['Close Price'].iloc[:492],
        'WIPRO' : wipro_data['Close Price'].iloc[:492],
        'ITC' : itc_data['Close Price'].iloc[:492],
        'AIRTEL' : airtel_data['Close Price'].iloc[:492] 
}
Close_price_data = pd.DataFrame(data,columns = ['JKLAKSHMI','RAYMOND','WIPRO','ITC','AIRTEL'])
print("Closing Prices of the 5 respective stocks")
print(Close_price_data)


# ### Query 5.3
# Prepare a scatter plot for differing weights of the individual stocks in the portfolio , the axes being the returns and volatility. Colour the data points based on the Sharpe Ratio ( Returns/Volatility) of that particular portfolio.

# In[ ]:


daily_returns = Close_price_data.pct_change()
daily_returns_mean = daily_returns.mean()
daily_returns_mean_reshaped = daily_returns_mean.values.reshape(5,1)
cov_matrix = daily_returns.cov()
weights = np.asarray([0.2,0.2,0.2,0.2,0.2])
portfolio_return = round(np.sum(daily_returns_mean_reshaped * weights) * 252,2)
portfolio_std_dev = round(np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(252),2)
print('Portfolio expected annualised return is {} and volatility is {}'.format(portfolio_return,portfolio_std_dev))

num_portfolios = 25000
results = np.zeros((3,num_portfolios))
for i in range(num_portfolios):
    weights = np.random.random(5)
    weights /= np.sum(weights)
    
    portfolio_returns = np.sum(daily_returns_mean * weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(252)
    
    results[0,i] = portfolio_returns
    results[1,i] = portfolio_std_dev
    results[2,i] = results[0,i] / results[1,i]    
results_frame = pd.DataFrame(results.T,columns=['ret','stdev','sharpe'])
print(results_frame)

plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.sharpe)
plt.colorbar()


# ### Query 5.4
# Mark the 2 portfolios where - Portfolio 1 - The Sharpe ratio is the highest Portfolio 2 - The volatility is the lowest.

# In[ ]:


cols = ['JKLAKSHMI','RAYMOND','WIPRO','ITC','AIRTEL']
resultsQuery4 = np.zeros((4+len(cols)-1,num_portfolios))
for i in range(num_portfolios):
    weights = np.random.random(5)
    weights /= np.sum(weights)    
    portfolio_return = np.sum(daily_returns_mean * weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(252)
    resultsQuery4[0,i] = portfolio_return
    resultsQuery4[1,i] = portfolio_std_dev
    resultsQuery4[2,i] = results[0,i] / results[1,i]  
    for j in range(len(weights)):
        resultsQuery4[j+3,i] = weights[j]
results_frameQuery4 = pd.DataFrame(resultsQuery4.T,columns=['Returns','Standard_Deviation','Sharpe_Ratio',cols[0],cols[1],cols[2],cols[3],cols[4]])
max_sharpe_port = results_frameQuery4.iloc[results_frameQuery4['Sharpe_Ratio'].idxmax()]
min_vol_port = results_frameQuery4.iloc[results_frameQuery4['Standard_Deviation'].idxmin()]


plt.scatter(results_frameQuery4.Standard_Deviation,results_frame.Returns,c=results_frame.Sharpe_Ratio)
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.legend('Red - Max')
plt.colorbar()
plt.scatter(max_sharpe_port[1],max_sharpe_port[0],marker=(5,1,0),color='red',s=700)
plt.scatter(min_vol_port[1],min_vol_port[0],marker=(5,1,0),color='yellow',s=700)

