#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import operator
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df_test = pd.read_csv(r'Please add a test.csv file location')


# In[3]:


df_test


# In[4]:


df_train = pd.read_csv(r'C:\Himanshu\MMSO\Sem - 2\Machine Learning\train.csv')


# In[5]:


df_train


# 1.) Removing the rows with NULL values

# In[6]:


df_test = df_test.dropna()
df_train = df_train.dropna()


# In[7]:


df_test


# In[8]:


df_train


# # Linear Regression: y = c + m * x, 
# # y = predicted value
# # x = input data
# # m = slope
# # c = Estimated regreddion ;ine crosses the y-axis

# In[9]:


x = np.array(df_train['x'])
y = np.array(df_train['y'])

mean_x = np.mean(x)
mean_y= np.mean(y)


# Calculating slope: m & c

# In[10]:


m = len(x)
Numerator = 0
Denominator = 0
for i in range(m):
    Numerator += (x[i] - mean_x) * (y[i] - mean_y)
    Denominator += (x[i] - mean_x) ** 2
m = Numerator/Denominator
c = mean_y - (m * mean_x)
print(f'm = {m} \nc = {c}')


# Calculating cost function J

# In[11]:


max_x = np.max(x) + 100
min_x = np.min(y) - 100

x = np.linspace (min_x, max_x, 100)
y = c + m * x


# In[12]:


plt.plot(x, y, color = '#6b58b9', label = 'Regression Line')
plt.scatter(x, y, color = '#b9585e', label = 'Data Points')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[16]:


Total_Sum_Square = 0
Rest_Sum_Square = 0

for i in range(int(100)):
    y_predict = c + m * x[i]
    Total_Sum_Square += (y[i] - mean_y) ** 2
    Rest_Sum_Square += (y[i] - y_predict) **2
R_square = 1 - (Rest_Sum_Square/Total_Sum_Square)
print(R_square)


# In[ ]:




