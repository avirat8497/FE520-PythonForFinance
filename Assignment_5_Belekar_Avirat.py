# Question 1

import numpy as np
import pandas as pd
import datetime as dt


# # Read data from an excel sheet

# In[2]:


data = pd.read_excel('Energy.xlsx')


# # Convert DataDate column to datetime format 

# In[3]:


data['Data Date'] = pd.to_datetime(data['Data Date'], format='%Y%m%d')


# # Get column number of Accumulated Other Comprehensive Income (Loss) and Selling, General and Administrative Expenses 

# In[4]:


s = data.columns.get_loc("Accumulated Other Comprehensive Income (Loss)")
d = data.columns.get_loc("Selling, General and Administrative Expenses")


# # Create a new data Frame with only one column ie.Data Date

# In[5]:


new_data = data.iloc[:,1]


# # Create a dataframe with the columns mentioned in the question

# In[6]:


new_data_1 = data.iloc[:,16:376]


# # Combine the two dataframe 

# In[7]:


frames = [new_data,new_data_1]
final_data = pd.concat(frames,axis = 1)
print(final_data.shape)


# # Display the final data Frame

# In[8]:


final_data


# # Add another column year to the dataframe 

# In[9]:


final_data['year'] = final_data['Data Date'].dt.year
final_data


# # Create a split function to split the data frame into training and testing based the conditions mentioned in the question 

# In[10]:


def split(start,end):
    if end == None:
        test = final_data[final_data['year'] == start]
        train = final_data[final_data['year'] != start]
        return train.to_numpy(),test.to_numpy()
    else:
        test = final_data[(final_data['year'] >= start) & (final_data['year'] <= end)]
        train = final_data[(final_data['year'] < start) | (final_data['year'] > end)]
        return train.to_numpy(),test.to_numpy()


# # Returning numpy array of the dataframe 

# In[11]:


print('Start Year = 2012, End year = None',split(2012,None))
print('Start Year = 2010, End year = 2013',split(2010,2013))


# In[ ]:


# Question 2

#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[46]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt


# # Reading the data using pd.read excel

# In[47]:


data = pd.read_excel('ResearchDatasetV2.0.xlsx',parse_dates= True)


# In[48]:


#data.head()


# In[49]:


data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')


# In[50]:


data.describe()


# In[51]:


y = data['Signal'].tolist()


# In[52]:


x = data['ClosePrice'].tolist()


# In[53]:


plt.plot(data['Date'].tolist(), y, color='r', label='signal')
plt.plot(data['Date'].tolist(), x, color='g', label='closeprice')
plt.show()


# # Calculating signal returns and price returns

# In[54]:



data['signal_return'] = data['Signal'].pct_change()
data['price_return'] = data['ClosePrice'].pct_change()
data.describe()


# # Plotting signal return and price return against time

# In[55]:



plt.plot(data['Date'].tolist(), data['signal_return'], color='r', label='signalreturn')
plt.plot(data['Date'].tolist(), data['price_return'], color='g', label='closepricereturn')
plt.title('Plot of Price and Signal returns vs Time')
plt.show()


# # Define a function that caluclates cross correlation for a given lag

# In[56]:


def cross_correlate(x, y, lag):
    return x.corr(y.shift(lag))


# # Find out cross correlation between price return and signal returns with given lags

# In[57]:


cross_corrs = []
for lag in range(1, 11):
    cross_corrs.append(cross_correlate(data['price_return'],
                                      data['signal_return'], lag))
plt.plot(range(1,11), cross_corrs, marker='o')
plt.xlabel('Lags')
plt.ylabel('Correlation')
plt.title('Correlation between price returns and signal returns with lag k')
plt.show()


# In[ ]:





# # The lag of 3 is significantly greater than the others. Therefore, we shall now use signals with a lag =3 for subsequent steps and trading strategies

# # Divide the signal returns into the required groups

# In[58]:

grp1 = []
grp1_idx = []
grp2 = []
grp2_idx = []
grp3 = []
grp3_idx = []
grp4 = []
grp4_idx = []
for index in range(len(data['signal_return'])):
    if data['signal_return'][index] < -0.01:
        grp1.append(data['signal_return'][index])
        grp1_idx.append(index)
    elif data['signal_return'][index] >= -0.01 and data['signal_return'][index] < 0:
        grp2.append(data['signal_return'][index])
        grp2_idx.append(index)
    elif data['signal_return'][index] >= 0 and data['signal_return'][index] < 0.01:
        grp3.append(data['signal_return'][index])
        grp3_idx.append(index)
    elif data['signal_return'][index] >= 0.01:
        grp4.append(data['signal_return'][index])
        grp4_idx.append(index)
grp1_idx = [x+3 for x in grp1_idx if x+3 < len(data.index)]
grp2_idx = [x+3 for x in grp2_idx if x+3 < len(data.index)]
grp3_idx = [x+3 for x in grp3_idx if x+3 < len(data.index)]
grp4_idx = [x+3 for x in grp4_idx if x+3 < len(data.index)]


# # Calculate the average returns within the signal groups

# In[59]:

average_returns = [np.mean(data['price_return'][grp1_idx]),
                  np.mean(data['price_return'][grp2_idx]),
                  np.mean(data['price_return'][grp3_idx]),
                  np.mean(data['price_return'][grp4_idx])]
grps = ['<-0.01', '-0.01:0', '0:0.01', '>0.01']


# In[60]:


average_returns


# In[61]:


plt.bar(grps, average_returns)
plt.title('Plot of average returns')
plt.xlabel('Groups of singal returns')
plt.ylabel('Average returns across groups')
plt.show()


# # Trading strategy

# In[62]:


# Start with $100 and assuming the portfolio is self financing
value = [100]
investment_available = 100
data['buy_signal'] = pd.Series(np.zeros(len(data.index)))
data['sell_signal'] = pd.Series(np.zeros(len(data.index)))
invested = False
for i in range(len(data.index)):
    
    if data['buy_signal'][i] == 1:
        shares = investment_available/data['ClosePrice'][i]
        invested = True
        
    if data['sell_signal'][i] == 1:
        investment_available = shares * data['ClosePrice'][i]
        
    if data['signal_return'][i] < 0 and invested == False:
        data['buy_signal'][i+3] = 1
        
        
    if invested == False:
        value.append(value[i-1])
        
    if invested == True:
        value.append(shares*data['ClosePrice'][i])
        
    if data['sell_signal'][i] == 1:
        invested = False
        
    if data['signal_return'][i] > 0.01 and invested == True:
        data['sell_signal'][i+3] = 1


# In[63]:


# Plotting the value of the investment on every date of trading
plt.plot(data['Date'], value[1:])
plt.xlabel('Date')
plt.ylabel('value of portfolio')
plt.title('Change in value of portfolio')
plt.show()


# In[ ]:





# In[64]:


# Total return of the trading strat
total_return = value[-1]/value[0] -1 
total_return


# In[65]:


# Plotting buy signals vs close prices
plt.plot(data['ClosePrice'], data['buy_signal'])


# The buy signal vs close price chart shows us that most of the buy signals are generated when the prices are low and thus allows the strategy to sell when prices are high and thus generating profits

# In[66]:


# Calculating sharpe ratio
risk_free = 0.01 #1% rate of return on risk free asset
portfolio_return = total_return # Calculated above
std = np.std(pd.Series(value).pct_change())


# In[67]:


sharpe = (total_return - risk_free) / std


# In[68]:


sharpe


# Question 3

#!/usr/bin/env python
# coding: utf-8

# # Import Numpy,Pandas and sklearn Libaries

# In[1]:


import pandas as pd
import numpy as np
from sklearn import datasets,linear_model


# # Load the diabetes dataset from sklearn,datasets and store it in data variable

# In[2]:


data = datasets.load_diabetes()


# # Store all features in X variable and target variable in y

# In[3]:


X = data.data
y = data.target
print(X.shape,y.shape)


# # Import train test split from sklearn

# In[4]:


from sklearn.model_selection import train_test_split


# # Split the data into 80% training and 20% testing 

# In[5]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20)


# # Display the shape of training and testing data

# In[6]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# # Import Linear Regression Model from sklearn

# In[7]:


model = linear_model.LinearRegression()


# # Fit the training data on the model created

# In[8]:


model.fit(X_train,y_train)


# # Predict the new value using X_test

# In[9]:


pred = model.predict(X_test)


# In[10]:


pred.shape


# In[11]:


from sklearn.metrics import r2_score


# # Printing coefficiant of the Linear Model 

# In[12]:


coefficient = model.coef_
print(coefficient)


# # Printing r2 score of the model

# In[13]:


r2_score = r2_score(y_test,pred)
print(r2_score)


# # Import cross validation score

# In[14]:


from sklearn.model_selection import cross_val_score


# # Number of fold is 10

# In[15]:


kFold = 10


# In[16]:


linearRegresssion = model.fit(X_train,y_train)
kRange = [1,2,3,4,5,6,7,8,9,10]


# # Fit the 10fold cross validation on the linear regression model 

# In[17]:


for kValue in kRange:
    value = cross_val_score(linearRegresssion,X,y,cv=10)
    print(value)


# In[18]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score


# # Create a Random Forest Regressor Model 

# In[19]:


regressor = RandomForestRegressor(max_depth = 7,random_state = 0)


# # Fit the model on the training data 

# In[20]:


regressor.fit(X_train,y_train)


# # Predict the number of trees and its branches using the model created 

# In[21]:


pred = regressor.predict(X_test)
print(pred)


# # Print Random Forest Regressor score 

# In[22]:


regressor.score(X_train,y_train)


# In[23]:


from sklearn.model_selection import GridSearchCV


# # Create parameters as mentioned in the question 

# In[24]:


parameters = {'max_depth': [None,7,4],'min_samples_split':[2,10,20]}


# # Create a model using grid search to esitmate the number of parameters 

# In[25]:


grid_GBR = GridSearchCV(estimator = regressor,param_grid = parameters,cv = 2,n_jobs = 1)


# # Fit the model on the train data

# In[26]:


grid_GBR.fit(X_train,y_train)


# In[27]:


print('Results from Grid Search')
print('\nThe best estimator across All params is:',grid_GBR.best_estimator_)
print('\nThe best score across All params is:',grid_GBR.best_score_)
print('\nThe best parameter across All params is:',grid_GBR.best_params_)















