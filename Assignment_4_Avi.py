#!/usr/bin/env python
# coding: utf-8

# # Question 1

# # Class for Linear Regression

# In[1]:


import numpy as np
class Linear_regression:

    def __init__(self, x, y, m, c, epochs, L):  # initilize variables
        self.x = x
        self.y = y
        self.m = m
        self.c = c
        self.epochs = epochs
        self.L = L

    def gradient_descent(self):  # calculate gradient descent
        x_trans = self.x.transpose()
        y_trans = self.y.transpose()
        ones = np.ones(np.shape(y))
        for i in range(self.epochs):
            xm = np.matmul(self.x,self.m)
            sum_xm = xm + self.c
            diff = y_trans - sum_xm
            #print(np.shape(diff))
            delta_m = -1 * np.matmul(x_trans,diff)
            delta_c = -1 * np.matmul(ones,diff)
            self.m = self.m - self.L * delta_m
            #print(np.shape(delta_c))
            self.c = self.c -  self.L * delta_c
            #print(np.shape(self.c))
        return self.m, self.c

    def predict(self, x_new):
        y_pred_new = []  
        y_new = np.dot(x_new,self.m) + self.c
        return y_new


# In[2]:


x = np.random.rand(100,10)
y = np.random.rand(1,100)
x_new = np.random.rand(30,10)
m = np.zeros((10,1))
c = 0
Linear_model = Linear_regression(x, y, m, c, 500, 0.001)
print("my m and c: ", Linear_model.gradient_descent())
print("my predict: ", Linear_model.predict(x_new))
print("my predict shape: ", np.shape(Linear_model.predict(x_new)))


# # Import Libraries like numpy and pandas

# In[3]:


import pandas as pd
import numpy as np
from pandas import DataFrame,Series


# # Create a function to replace characters and clean 

# In[4]:


def cleaning(dataframe):
    data = dataframe['Amount'].astype(str)
    data = data.map(lambda x: x.replace('$','').replace(')','').replace('zero',''))
    data = data.map(lambda x: x.replace('(','-'))
    data = data.astype(float)
    data = data.sum().round(2)
    return data


# # Import the dataset

# In[5]:


new_data = pd.DataFrame(pd.read_csv('res_purchase_2014.csv'))


# # Calculate the total amount

# In[6]:


amount = cleaning(new_data)
print('Full total = ',amount)


# # Calculate amount for Grainger

# In[7]:


grainger = new_data.loc[new_data['Vendor'] == 'WW GRAINGER']
print('Grainger Total = ',cleaning(grainger))


# # Calculate amount for SuperCenter

# In[8]:


supercenter = new_data.loc[new_data['Vendor'] == 'WM SUPERCENTER']
print('SuperCenter Total = ',cleaning(supercenter))


# # Calculate amount for Grocery Store Total

# In[9]:


grocery = new_data.loc[new_data['Merchant Category Code (MCC)'] == 'GROCERY STORES,AND SUPERMARKETS']
print('Grocery Store Total = ',cleaning(grocery))


# # Question 3

# # Read the dataset

# In[10]:


BS = pd.read_excel('Energy.xlsx')
Rating = pd.read_excel('EnergyRating.xlsx')
print(BS.shape)
print(Rating.shape)


# # Remove Columns with more than 30% missing values

# In[11]:


limit_BS = BS.shape[1] * 0.3


# In[12]:


BS = BS.dropna(axis =1,thresh = limit_BS)
print(BS.shape)


# # Remove columns with more than 90% of the values

# In[13]:


BS = BS.drop(columns = BS.columns[((BS == 0).mean()>0.9)],axis = 1)
print(BS.shape)


# # Replace the NaN values with the mean

# In[14]:


BS.replace('',np.nan)
BS = BS.fillna(BS.mean())
print(BS)


# # Normalize the data

# In[15]:


new_BS = BS.loc[:,'Accumulated Other Comprehensive Income (Loss)':
                                   'Selling, General and Administrative Expenses'].apply(lambda x:(x-x.min()) / (x.max() - x.min()))
print(new_BS)


# # Evaluate the correaltion matrix

# In[16]:


corr_val = new_BS[['Current Assets - Other - Total',
    'Current Assets - Total',
    'Other Long-term Assets',
    'Assets Netting & Other Adjustments']]
print(corr_val.corr())


# # Merge balance Sheet and Rating using inner join 

# In[17]:


Matched = BS.merge(Rating,on = ['Data Date','Global Company Key'],how = 'inner')
print(Matched.shape)
print(Matched)


# # Creating a new map function that maps ratings to numerical value

# In[18]:


def rating(psn):
    if psn == 'AAA':
        return 1
    elif psn == 'AA+':
        return 2
    elif psn == 'AA':
        return 3
    elif psn == 'AA-':
        return 4
    elif psn == 'A+':
        return 5
    elif psn == 'A-':
        return 6
    elif psn == 'BBB+':
        return 7
    elif psn == 'BBB':
        return 8
    elif psn == 'BBB-':
        return 9
    elif psn == 'BB+':
        return 10
    elif psn == 'BB':
        return 11
    elif psn == 'others':
        return 12


# In[19]:


Matched['Rate'] = Matched['S&P Domestic Long Term Issuer Credit Rating'].astype(str).map(lambda x:rating(x))
print(Matched.head(5))


# In[20]:


Matched_correlation = Matched.corr()
Matched_correlation


# In[21]:


Matched_corr_rate = (Matched_correlation["Rate"])
Matched_corr_rate = np.abs(Matched_corr_rate)


# In[22]:


Matched_corr_rate.dropna(inplace = True)
Matched_corr_rate = Matched_corr_rate.sort_values()
final = Matched_corr_rate.tail(11)
print(final)


# In[23]:


final = final.head(10)
final


# In[24]:


index = ["Common Shares Used to Calc Earnings Per Share - Fully Diluted - 12 Months Moving",
            "Current Liabilities - Total",
            "Invested Capital - Total - Quarterly",
            "Liabilities and Stockholders Equity - Total",
            "Assets - Total",
            "Stockholders Equity - Total",
            "Common/Ordinary Equity - Total",
            "Stockholders Equity > Parent > Index Fundamental > Quarterly",
            "Receivables - Total",
            "Current Assets - Total"]


# In[25]:


y_new = Matched["Rate"]
x_new = Matched[index]
y_new.shape


# In[26]:


m = np.zeros((10,1))
c = 0


# In[27]:


Linear_model = Linear_regression(x_new, y_new, m, c, 500, 0.001)


# In[28]:


print("my m and c: ", Linear_model.gradient_descent())


# In[ ]:




