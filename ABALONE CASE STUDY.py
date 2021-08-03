#!/usr/bin/env python
# coding: utf-8

# In[50]:


#A place for the imports
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# # Data exploration

# In[13]:


df = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset1/master/abalone.csv')
df.head()


# In[14]:


df.info()


# #There are 8 numerical not-null features in the data. Feature Sex will need to be changed to dummy values in data preparation in order to use it in the model.
# 
# Let's investigate further the data as there is a possibility that some of the values that are not null are set to 0 instead.
# 
# 

# In[15]:


df.describe()


# As mentioned it seems that there are minimum values in Height that are 0

# In[17]:


df[df.Height == 0]


# There are two records where Height is equal to 0, it is possible that it was hard to measure it or it was simply omitted. Nevertheless, this can be treated as a NULL value and since there are only two records like that it will be simplest to ignore them.

# In[18]:


df = df[df.Height > 0]
df.describe()


# In[19]:


df.hist(figsize=(20,10), grid = False, layout=(2,4), bins = 30);


# Histograms show that the data may be skewed, so it will be reasonable to measure it.
# 
# It also shows that there are possible outliers in Height and that there might be a strong relationship between the Diameter and Lenght and between Shell weight, Shucked weight Viscera weight and Whole weight.
# 
# 

# In[20]:


nf = df.select_dtypes(include=[np.number]).columns
cf = df.select_dtypes(include=[np.object]).columns


# In[21]:


skew_list = stats.skew(df[nf])
skew_list_df = pd.concat([pd.DataFrame(nf,columns=['Features']),pd.DataFrame(skew_list,columns=['Skewness'])],axis = 1)
skew_list_df.sort_values(by='Skewness', ascending = False)


# In[31]:


sns.set()
cols = ['Length','Diameter','Height','Whole weight', 'Shucked weight','Viscera weight', 'Shell weight','Rings']
sns.pairplot(df[cols], height = 2.5)
plt.show();


# Observations:
# 
# - Many features are highly correlated
#     - length and diameter show linear correlation
#     - the length and weight features are quadratic correlated
#     - whole weight is linearly correlated with other weight features
# - Number of Rings is positively corelated with almost all quadratic features
# - Possible outliers in Height features
# 
# Scatter plot analysis also shows that data mostly cover the values for Rings from 3 to little over 20, selecting only this data in the model may be taken under consideration to increase the accuracy.
# 
# First I will take a closer look at the Height outliers and then I will investigate correlations between the features.

# In[32]:


data = pd.concat([df['Rings'], df['Height']], axis = 1)
data.plot.scatter(x='Height', y='Rings', ylim=(0,30));


# Two values seem not to follow the trend, that is why I will treat them as outliers and delete from data

# In[33]:


df = df[df.Height < 0.4]
data = pd.concat([df['Rings'], df['Height']], axis = 1)
data.plot.scatter(x='Height', y='Rings', ylim=(0,30));


# In[34]:


df.hist(column = 'Height', figsize=(20,10), grid=False, layout=(2,4), bins = 30);


# Deleted data as suspected was the cause for the skewness of Height feature, now it is closer to a normal distribution.

# # Correlation matrix

# In[36]:


corrmat = df.corr()
cols = corrmat.nlargest(8, 'Rings')['Rings'].index
cm = np.corrcoef(df[nf].values.T)
sns.set(font_scale=1.25)
plt.figure(figsize=(15,15))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=nf.values, xticklabels=nf.values)
plt.show();


# # Categorical Feature
# I will analyse the relation of Rings with the Sex feature

# In[37]:


data = pd.concat([df['Rings'], df['Sex']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxenplot(x='Sex', y="Rings", data=df)
fig.axis(ymin=0, ymax=30);


# Distribution between Male and Female is similar
# Most of the Rings both for Male and Female are between 8 and 19
# Infants have mostly from 5 to 10 Rings
# The plot also shows that Rings majority lies between 3 to 22, as mentioned previously.

# # Linear Regression Models

# In[38]:


df = pd.get_dummies(df)
df.head()


# Now I will set the X and y labels

# In[39]:


X = df.drop(['Rings'], axis = 1)
y = df['Rings']


# In[40]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.3)


# In[41]:


from sklearn.linear_model import LinearRegression 
paramLin = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
LinearReg = GridSearchCV(LinearRegression(),paramLin, cv = 10)
LinearReg.fit(X = X_train,y= y_train)
LinearRegmodel = LinearReg.best_estimator_
print(LinearReg.best_score_, LinearReg.best_params_)


# In[42]:


LinearReg.score(X_train,y_train)


# In[43]:


LinearReg.score(X_test,y_test)


# # Random Forest

# In[48]:


rf_classifier = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_features='auto', n_estimators=30, oob_score=False,n_jobs=-1)
rf_classifier.fit(X_train, y_train)


# In[51]:


rf_classifier_train = rf_classifier.predict(X_train)
accuracy_score(y_train, rf_classifier_train)


# In[ ]:




