#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 


# In[3]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 


# In[4]:


from imblearn import under_sampling, over_sampling
from sklearn.linear_model import LogisticRegression


# In[5]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix


# In[6]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


# In[11]:


data = pd.read_csv("Desktop\Solar Business\solar features.csv")


# In[16]:


data.head()
print('There are', str(len(data)), 'rows of data in this dataset')
print('There are', str(data.shape[1]), 'features in this dataset')
data.head(15)


# In[13]:


#Check Dataset 
data.info()


# In[14]:


data.describe()


# In[17]:


data['has_collaboration'].value_counts()


# In[18]:


data['business_model_type'].value_counts()


# In[19]:


data['is_automotive'].value_counts()


# In[20]:


data['is_energy'].value_counts()


# In[21]:


data['is_healthcare'].value_counts()


# In[23]:


data['industry_knowledge'].value_counts()


# In[ ]:




