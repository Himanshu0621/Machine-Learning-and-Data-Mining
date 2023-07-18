#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from itertools import groupby 

df = pd.read_csv('C:\Himanshu\MMSO\Sem - 2\Machine Learning\iris.csv')


# In[2]:


df


# In[8]:


# Solution to part 1
def pearsonCorr(x,y):
    d = { 'X' : x , 'Y' : y}
    df = pd.DataFrame(data=d)
    correlation = df.corr()
    return correlation.loc['X', 'Y']

pearson_Corr_coef = print(pearsonCorr(df['sepal_length'],df['sepal_width'] ))
pearson_Corr_coef


# In[9]:


#Solution to part 2
import seaborn as sns
sns.scatterplot(x="sepal_length", y="sepal_width", data=df);


# In[10]:


#Solution to part 2
dfmat = df.corr()
dfmat


# In[11]:


#Solution to part 2
sns.heatmap(dfmat);


# In[12]:


#Solution to part 3
import scipy.stats as stats
df['sepal_length zscore'] = stats.zscore(df['sepal_length'])
df['sepal_width zscore'] = stats.zscore(df['sepal_width'])
df['petal_length zscore'] = stats.zscore(df['petal_length'])
df['petal_width zscore'] = stats.zscore(df['petal_width'])
df


# In[22]:



# Solution to part 4
Zscore_df = df.loc[:,'sepal_length zscore':]
Zscore_df


# In[24]:


Zscore_dfmat = Zscore_df.corr()
Zscore_dfmat


# In[29]:


#Solution to part 5
from numpy.linalg import eig
a = np.array(dfmat)
w,v=eig(a)
print('E-value:', w)
print('E-vector', v)


# In[ ]:




