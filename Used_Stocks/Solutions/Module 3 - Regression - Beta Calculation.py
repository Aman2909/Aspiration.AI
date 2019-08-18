
# coding: utf-8

# # Module 3- Regression & Beta Calculation
# 
# 

#    ### Welcome to the Answer notebook for Module 3 ! 
# Make sure that you've submitted the module 2 notebook and unlocked Module 3 yourself before you start coding here
# 

# #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ### Query 3.1 
# Import the file 'gold.csv', which is contains the data about last 2 years price action of Indian(MCX) gold standard. Explore the dataframe. You'd see 2 unique columns - 'Pred' and 'new'.
# 
# One of the 2 columns is a linear combination of the OHLC prices with varying coefficients while the other is a polynomial fucntion of the same inputs. Also, one of the 2 columns is partially filled.
# 
# >Using linear regression, find the coefficients of the inputs and using the same trained model, complete the
#       entire column.
#       
# >Also, try to fit the other column as well using a new linear regression model. Check if the predicitons are 
#       accurate.
#       Mention which column is a linear function and which is a polynomial function.
#       (Hint: Plotting a histogram & distplot helps in recognizing the  discrepencies in prediction, if any.)

# In[ ]:


import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


Data_Gold = pd.read_csv("GOLD.csv")
Data_Gold_non_missing = Data_Gold.dropna()

x = np.array(Data_Gold_non_missing["new"])
y = np.array(Data_Gold_non_missing["Pred"])
x = x.reshape(-1,1)
y = y.reshape(-1,1)
regression_model = LinearRegression()
regression_model.fit(x, y)
y_predicted = regression_model.predict(x)
rmse = mean_squared_error(y, y_predicted)
r2 = r2_score(y, y_predicted)
print('Slope:' ,regression_model.coef_,', Intercept:', regression_model.intercept_,', Root mean squared error: ', rmse,', R2 score: ', r2)
plt.scatter(x, y, s=10, color = 'g')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, y_predicted, color='r')
plt.show()

Data_Gold_Complete = Data_Gold[:]
X = Data_Gold['new']
X = X.values.reshape(-1,1)
Y_pred = regression_model.predict(X)
Y_pred = pd.Series(Y_pred.ravel())
Y_pred = Y_pred.to_frame()
Data_Gold_Complete['Pred'] = Y_pred

x = np.array(Data_Gold_Complete["Pred"])
y = np.array(Data_Gold_Complete["new"])
x = x.reshape(-1,1)
y = y.reshape(-1,1)
regression_model = LinearRegression()
regression_model.fit(x, y)
y_predicted = regression_model.predict(x)
rmse = mean_squared_error(y, y_predicted)
r2 = r2_score(y, y_predicted)
print('Slope:' ,regression_model.coef_,', Intercept:', regression_model.intercept_,', Root mean squared error: ', rmse,', R2 score: ', r2)
plt.scatter(x, y, s=10, color = 'g')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, y_predicted, color='r')
plt.show()

plt.hist(Data_Gold_Complete['Pred'])
plt.show()
sns.distplot(Data_Gold_Complete['Pred'])
plt.show()


# ### Query 3.2
# Import the stock of your choosing AND the Nifty index. 
# Using linear regression (OLS), calculate -
# The daily Beta value for the past 3 months. (Daily= Daily returns)
# The monthly Beta value. (Monthly= Monthly returns)
# Refrain from using the (covariance(x,y)/variance(x)) formula. 
# Attempt the question using regression.(Regression Reference) 
# Were the Beta values more or less than 1 ? What if it was negative ? 
# Discuss. Include a brief writeup in the bottom of your jupyter notebook with your inferences from the Beta values and regression results

# In[ ]:


Data_Gold_Complete.set_index('Date',inplace = True)

Data1 = pd.read_csv('CENTURYPLY.csv')
Data1['Date'] = pd.to_datetime(Data1['Date'])
Data1 = Data1.sort_values('Date')
Data1.set_index('Date', inplace=True)

Nifty = pd.read_csv('NIFTY50_Data.csv')
Nifty['Date'] = pd.to_datetime(Nifty['Date'])
Nifty = Nifty.sort_values('Date')
Nifty.set_index('Date', inplace=True)

plt.figure(figsize=(20,10))
fil_data1 = Data1.iloc[37:]
fil_nifty = Nifty.iloc[:457]
fil_data1 = fil_data1['Close Price'].pct_change()
fil_nifty = fil_nifty['Close'].pct_change()
fil_data1.plot()
fil_nifty.plot()
plt.ylabel("Daily Return of CENTURYPLY and NIFTY")
plt.show()

fil_data1['pct_change'] = Data1.iloc[37:]['Close Price'].pct_change()
fil_nifty['pct_change'] = Nifty.iloc[:457]['Close'].pct_change()
x = fil_data1['pct_change'].dropna()
y = fil_nifty['pct_change'].dropna()
myModel = sm.OLS(y,x).fit()
myModel.summary()

tcs = pd.read_csv('TCS.NS.csv', parse_dates=True, index_col='Date',)
nifty50 = pd.read_csv('^NSEI.csv', parse_dates=True, index_col='Date')

monthly_prices = pd.concat([tcs['Close'], nifty50['Close']], axis=1)
monthly_prices.columns = ['TCS', 'NIFTY50']
print(monthly_prices.head())
monthly_returns = monthly_prices.pct_change(1)
clean_monthly_returns = monthly_returns.dropna(axis=0)
print(clean_monthly_returns.head())


X = clean_monthly_returns['TCS']
y = clean_monthly_returns['NIFTY50']
X1 = sm.add_constant(X)
model = sm.OLS(y, X1)
results = model.fit()
print(results.summary())

# Beta value for TCS is less than 1 i.e. 0.1327