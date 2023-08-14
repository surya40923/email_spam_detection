#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[28]:


df=pd.read_csv('spam.csv', encoding='latin-1')
df.head()


# In[54]:


df['spam']=df['v1'].replace({'ham':0,'spam':1})


# In[39]:


df['v2'].values[2]


# In[29]:


from sklearn.model_selection import train_test_split


# In[55]:


x_train,x_test,y_train,y_test=train_test_split(df.v2,df.spam,test_size=0.2)


# In[56]:


from sklearn.feature_extraction.text import CountVectorizer


# In[65]:


c=CountVectorizer()
v=c.fit_transform(x_train.values)
v.toarray()[:3]


# In[58]:


from sklearn.naive_bayes import MultinomialNB


# In[59]:


mb=MultinomialNB()
mb.fit(v,y_train)


# In[60]:


e_count=c.fit_transform(x_train.values)
t=mb.predict(e_count)
t[1]


# In[ ]:




