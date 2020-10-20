#!/usr/bin/env python
# coding: utf-8

# #                                                                                                 NIDHI YADAV

# ## GRIP

# ### Task 1 : Simple Linear Rigression

# To predict the percentage of marks scored by a student based upon the number of study hours.

#       

# #### importing libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Reading Data
Url="http://bit.ly/w-data"
df=pd.read_csv(Url)
df.head()


# In[4]:


df.describe()


# ##### Plotting graph

# In[5]:


df.plot(x='Hours', y='Scores',style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# ###### The above graph shows, there is a positive linear relation between the number of hours studied and the percentage of score.

# ## Preparing Data

# In[6]:


# Dividng the data into "attributes" and "labels" i.e. inputs and outputs
X = df.iloc[:, :-1].values  
y = df.iloc[:, 1].values  


# ###### Splitting the data into train set and test set using SciKit Learn and then training our Algorithm.

# In[7]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# In[16]:


# Shape of training Data
print(X_train.shape)
print(y_train.shape)


# In[8]:


from sklearn.linear_model import LinearRegression  
model_LR= LinearRegression()  
model_LR.fit(X_train, y_train) 


# In[9]:


# predicting with test data
print(X_test)
y_pred=model_LR.predict(X_test)
print('predicted values of y are : ',y_pred)


# In[10]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[11]:


# Plotting the regression line
line = (model_LR.coef_*X)+model_LR.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# ###### Predicted score if a student studies for 9.25 hrs/day

# In[14]:


hours = 9.25
pred = model_LR.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(pred[0]))


# In[17]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# ### CONCLUSION : The percentage of marks scored by the student who studied for 9.25 hours is 93.69%

# In[ ]:




