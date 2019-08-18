
# coding: utf-8

# # Module 2- Plotting in Financial Markets
# 

#    ### Welcome to the Answer notebook for Module 2 ! 
# Make sure that you've submitted the module 1 notebook and unlocked Module 2 yourself before you start coding here
# 

# #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ### Query 2.1 
# Load the week2.csv file into a dataframe. What is the type of the Date column? Make sure it is of type datetime64. Convert the Date column to the index of the dataframe.
# Plot the closing price of each of the days for the entire time frame to get an idea of what the general outlook of the stock is.
# 
# >Look out for drastic changes in this stock, you have the exact date when these took place, try to fetch the news for this day of this stock
# 
# >This would be helpful if we are to train our model to take NLP inputs.

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from collections import Counter
warnings.filterwarnings('ignore')

data = pd.read_csv("Week2.csv")
print(data['Date'].dtypes)
data['Date'] = data['Date'].astype('datetime64[ns]')
print(data['Date'].dtypes)

plt.plot(data['Date'],data['Close Price'])
plt.show()


# ### Query 2.2
# A stem plot is a discrete series plot, ideal for plotting daywise data. It can be plotted using the plt.stem() function.
# 
# Display a stem plot of the daily change in of the stock price in percentage. This column was calculated in module 1 and should be already available in week2.csv. Observe whenever there's a large change.

# In[ ]:


plt.stem(data['Date'],data['Day_Perc_Change'])
plt.show()


# ### Query 2.3
# Plot the daily volumes as well and compare the percentage stem plot to it. Document your analysis of the relationship between volume and daily percentage change. 

# In[ ]:


plt.stem(data['Date'],data['Day_Perc_Change'])
plt.show()
plt.stem(data.Date,data['Total Traded Quantity'])
plt.show()
plt.stem(data.Date,data['Day_Perc_Change'])
plt.show()
plt.stem(data['Day_Perc_Change'],data['Total Traded Quantity'])
plt.show()
plt.plot(data['Day_Perc_Change'],data['Total Traded Quantity'])
plt.show()


# ### Query 2.4
# We had created a Trend column in module 1. We want to see how often each Trend type occurs. This can be seen as a pie chart, with each sector representing the percentage of days each trend occurs. Plot a pie chart for all the 'Trend' to know about relative frequency of each trend. You can use the groupby function with the trend column to group all days with the same trend into a single group before plotting the pie chart. From the grouped data, create a BAR plot of average & median values of the 'Total Traded Quantity' by Trend type.

# In[ ]:


trends_list = data['Trend'].tolist()
count = Counter(trends_list)
pie_chart_values = list(count.values())
legends = list(count.keys())
plt.pie(pie_chart_values,labels=legends)
plt.show()

data.groupby(['Trend'])['Total Traded Quantity'].mean().plot.bar()
data.groupby(['Trend'])['Total Traded Quantity'].median().plot.bar()


# ### Query 2.5
# Plot the daily return (percentage) distribution as a histogram.
# Histogram analysis is one of the most fundamental methods of exploratory data analysis. In this case, it'd return a frequency plot of various values of percentage changes.

# In[ ]:


plt.hist(data['Day_Perc_Change'])
plt.show()


# ### Query 2.6
# We next want to analyse how the behaviour of different stocks are correlated. The correlation is performed on the percentage change of the stock price instead of the stock price.
# 
# Load any 5 stocks of your choice into 5 dataframes. Retain only rows for which ‘Series’ column has value ‘EQ’. Create a single dataframe which contains the ‘Closing Price’ of each stock. This dataframe should hence have five columns. Rename each column to the name of the stock that is contained in the column. Create a new dataframe which is a percentage change of the values in the previous dataframe. Drop Nan’s from this dataframe.
# Using seaborn, analyse the correlation between the percentage changes in the five stocks. This is extremely useful for a fund manager to design a diversified portfolio. To know more, check out these resources on correlation and diversification. 

# In[ ]:


data1 = pd.read_csv('ASHOKA.csv')
data1 = data1[data1['Series']=='EQ']

data2 = pd.read_csv('BAJAJELEC.csv')
data2 = data2[data2['Series']=='EQ']

data3 = pd.read_csv('BOMDYEING.csv')
data3 = data3[data3['Series']=='EQ']

data4 = pd.read_csv('CENTURYPLY.csv')
data4 = data4[data4['Series']=='EQ']

data5 = pd.read_csv('FORTIS.csv')
data5 = data5[data5['Series']=='EQ']

cols = ['ASHOKA','BAJAJELEC','BOMDYEING','CENTURYPLY','FORTIS']
close_price_all = pd.DataFrame(columns=cols)
close_price_all['ASHOKA'] = data1['Close Price']
close_price_all['BAJAJELEC'] = data2['Close Price']
close_price_all['BOMDYEING'] = data3['Close Price']
close_price_all['CENTURYPLY'] = data4['Close Price']
close_price_all['FORTIS'] = data5['Close Price']
close_price_all=close_price_all.fillna(0)

pct_change= close_price_all.pct_change()
sns.set(color_codes=True)
sns.pairplot(pct_change)


# ### Query 2.7
# Volatility is the change in variance in the returns of a stock over a specific period of time.Do give the following documentation on volatility a read.
# You have already calculated the percentage changes in several stock prices. Calculate the 7 day rolling average of the percentage change of any of the stock prices, then compute the standard deviation (which is the square root of the variance) and plot the values.
# Note: pandas provides a rolling() function for dataframes and a std() function also which you can use.

# In[ ]:


rolling_ASHOKA = pct_change['ASHOKA'].rolling(7).mean()
print("Volatility =", rolling_ASHOKA)
standard_ASHOKA = pct_change['ASHOKA'].std()
print("Standard Deviation = ",standard_ASHOKA)

temp1 = pd.to_datetime(data1['Date'])
temp1 = temp1.tolist()
plt.plot(temp1,rolling_ASHOKA.tolist())
plt.show()


# ### Query 2.8
# Calculate the volatility for the Nifty index and compare the 2. This leads us to a useful indicator known as 'Beta' ( We'll be covering this in length in Module 3) 

# In[ ]:


nifty = pd.read_csv('NIFTY50_Data.csv')
nifty_close_price = nifty['Close']
nifty_change = nifty_close_price.pct_change().fillna(0).rolling(7).mean().fillna(0)
niftyDate = pd.to_datetime(nifty['Date'])
niftyDate = niftyDate.tolist()

data5_Date = pd.to_datetime(data5['Date'])
data5Lis = data5_Date.tolist()
data5_close_price = data5['Close Price']
data5_change = data5_close_price.pct_change().fillna(0).rolling(7).mean().fillna(0)

plt.title("Volatility of NIFTY with respect to ASHOKA and FORTIS")
plt.plot(niftyDate,nifty_change.tolist(),label = 'NIFTY')
plt.plot(temp1,rolling_ASHOKA.fillna(0).tolist(),label = 'ASHOKA')
plt.plot(data5Lis,data5_change,label = 'FORTIS')
plt.legend(loc='bottom')
plt.show()


# ### Query 2.9
# Trade Calls - Using Simple Moving Averages. Study about moving averages here. 
#  
# Plot the 21 day and 34 day Moving average with the average price and decide a Call ! 
# Call should be buy whenever the smaller moving average (21) crosses over longer moving average (34) AND the call should be sell whenever smaller moving average crosses under longer moving average. 
# One of the most widely used technical indicators.

# In[ ]:


plt.plot(data5Lis,data5_change,label = 'FORTIS')
plt.legend(loc='bottom')
plt.show()

short_window = 21
long_window = 34
signals = pd.DataFrame(index=data5.index)
signals['signal'] = 0.0
signals['short_mavg'] = data5['Close Price'].rolling(window=short_window, min_periods=1,center=False).mean()
signals['long_mavg'] = data5['Close Price'].rolling(window=long_window,min_periods=1, center=False).mean()
signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0,0.0)
signals['positions'] = signals['signal'].diff()
print(signals)

figbig = plt.figure(figsize=(20,15))
graph = figbig.add_subplot(111, ylabel='Price')
data5['Close Price'].plot(ax=graph, color='black', lw=2.)
signals[['short_mavg', 'long_mavg']].plot(ax=graph, lw=2.)
graph.plot(signals.loc[signals.positions == 1.0].index, signals.short_mavg[signals.positions == 1.0])
graph.plot(signals.loc[signals.positions == -1.0].index, signals.short_mavg[signals.positions == -1.0])
plt.show()


# ### Query 2.10
# Trade Calls - Using Bollinger Bands 
# Plot the bollinger bands for this stock - the duration of 14 days and 2 standard deviations away from the average 
# The bollinger bands comprise the following data points- 
# The 14 day rolling mean of the closing price (we call it the average) 
# Upper band which is the rolling mean + 2 standard deviations away from the average. 
# Lower band which is the rolling mean - 2 standard deviations away from the average. 
# Average Daily stock price.
# Bollinger bands are extremely reliable , with a 95% accuracy at 2 standard deviations , and especially useful in sideways moving market. 
# Observe the bands yourself , and analyse the accuracy of all the trade signals provided by the bollinger bands. 
# Save to a new csv file. 

# In[ ]:


symbol = 'FORTIS'
data_fortis = pd.read_csv('FORTIS.csv'.format(symbol), index_col='Date',
                 parse_dates=True, usecols=['Date', 'Close Price'],
                 na_values='nan')
data_fortis = data_fortis.rename(columns={'Close Price': symbol})
data_fortis.dropna(inplace=True)
sma = data_fortis.rolling(window=14).mean()
rstd = data_fortis.rolling(window=14).std()

upper_band = sma + 2 * rstd
upper_band = upper_band.rename(columns={symbol: 'upper'})
lower_band = sma - 2 * rstd
lower_band = lower_band.rename(columns={symbol: 'lower'})
data_fortis = data_fortis.join(upper_band).join(lower_band)
ax = data_fortis.plot(title='{} Price and BB'.format(symbol))
ax.fill_between(data_fortis.index, lower_band['lower'], upper_band['upper'], color='#ADCCFF', alpha='0.4')
ax.set_xlabel('Date')
ax.set_ylabel('SMA and BB')
ax.grid()
plt.show()

