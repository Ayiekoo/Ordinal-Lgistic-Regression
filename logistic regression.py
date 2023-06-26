#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import datasets
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans
from matplotlib import style
from scipy.cluster.hierarchy import dendrogram, linkage
style.use('ggplot')
from sklearn import preprocessing
import math


# In[46]:


import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('autosave', '1')


# In[47]:


# ordinal logistic regression

## IMPORT DATASET
df = pd.read_csv("C:/Users/Ayieko/Documents/MACHINE LEARNING TUTORIALS\ML\LOGISTIC REGRESSION/archive/winequality-red.csv")
print(df)


# In[48]:


df.head(13)


# In[49]:


df.describe


# In[50]:


df.info()


# In[51]:


sns.countplot(df['quality']) ## counts plot for quality


# In[52]:


df['quality'].value_counts()


# In[53]:


## Data visualization

df.info()


# In[54]:





# In[15]:


fig, ax = plt.subplots(3, 4, figsize=(20, 10))
for value, subplot in zip(range(0, 11), ax.flatten()): 
    sns.barplot(x=DataFrame['quality'], y=DataFrame[DataFrame.columns[value]], ax=subplot)
#    sns.regplot(x=df[df.columns[value]], y=df['quality'],ax=subplot,truncate=True,scatter=False)

fig.tight_layout()
plt.show()


# In[16]:


DataFrame['quality'].value_counts()


# In[17]:


fig, ax = plt.subplots(3, 4, figsize=(20, 10))
for value, subplot in zip(range(0,11), ax.flatten()):
    sns.barplot(x=DataFrame['quality'], y=DataFrame[DataFrame.columns[value]], ax=subplot)
      #sns.regplot(x=Data_Frame[Data_Frame.columns[value]], y=Data_Frame['quality'], ax=subplot, truncate=True, scatter=False)
fig.tight_layout()
plt.show()


# In[36]:


sns.pairplot(Data_Frame, hue='quality')
plt.show()


# In[18]:


df1 = DataFrame[['volatile acidity', 'quality']].copy()
df1.head()


# In[19]:


col = list(DataFrame)
col=col[0:11]


# In[55]:


for i in col:
    df1 = DataFrame[[i, 'quality']].copy()
    sns.pairplot(df1, hue='quality',size=5)
plt.show()


# df_corr=df.corr()
# df_corr

# In[82]:


df_corr=df.corr()
df_corr


# In[56]:


df.head(10)


# In[57]:


col_list = df.columns.values.tolist()
col_list


# In[58]:


df2=df.copy()
df2


# In[50]:


df2.drop(['quality'], axis='columns', inplace=True)
df2


# In[83]:


col_list2 = df2.columns.values.tolist()
col_list2


# In[51]:


col_list2 = df2.columns.values.tolist()
col_list2


# In[61]:


for i in col:
    df1 = DataFrame[[i, 'quality']].copy()
    sns.pairplot(df1, hue='quality',size=5)
plt.show()


# In[62]:


df2=df.copy()
print(df2)


# In[64]:


df2.head()


# In[84]:


### let's do a correlation heatmap

mask=np.array(df_corr)
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(df_corr,annot=True,cbar=True,square=True)


# In[ ]:


### the heatmap does not yield high correlations. the dataset remains as given


# In[42]:


####   ORDINAL LOGISTIC REGRESSION
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import statsmodels.api as sm


# In[ ]:





# In[68]:


df34=DataFrame.copy()
count34=0
for i in range (0,1599):
    if(df['quality'].iloc[i]>=4):
        df34['quality'].iloc[i]=1
        
        
    else:
        df['quality'].iloc[i]=0
        count34=count34+1


# In[73]:


count34


# In[69]:


df45=df.copy()
count45=0
for i in range (0, 1599):
    if (df['quality'].iloc[i]>=5):
        df45['quality'].iloc[i]=1
        
    else:
        df45['quality'].iloc[i]=0
        count45=count45+1


# In[74]:


count45


# In[71]:


df56=df.copy()
count56=0
for i in range (0, 1599):
    if (df['quality'].iloc[i]>=6):
        df56['quality'].iloc[i]=1
        
    else:
        df56['quality'].iloc[i]=0
        count=count56+1


# In[76]:


count56


# In[77]:


df67=df.copy()
count67=0
for i in range (0, 1599):
    if (df['quality'].iloc[i]>=7):
        df67['quality'].iloc[i]=1
        
        
    else:
        df67['quality'].iloc[i]=0
        count67=count67+1


# In[78]:


count67


# In[80]:


df78=df.copy()
count78=0
for i in range (0, 1599):
    if (df['quality'].iloc[i]>=8):
        df78['quality'].iloc[i]=1
        
    else:
        df78['quality'].iloc[i]=0
        count78=count78+1


# In[81]:


count78


# In[25]:


# multinominal regresssion
## multinominal vs. OVR


# In[ ]:





# In[40]:





# In[41]:





# In[29]:





# In[16]:





# In[ ]:




