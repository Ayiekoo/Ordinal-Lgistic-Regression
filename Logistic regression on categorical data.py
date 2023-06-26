#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


# Hypothetical data
data = pd.DataFrame({
    'Respondent': range(1, 11),
    'Opinion': ['Disagree', 'Strongly Disagree', 'Agree', 'Disagree', 'Strongly Agree',
                'Agree', 'Strongly Disagree', 'Disagree', 'Agree', 'Disagree']
})

# Display the dataset
print(data)


# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


# Define the mapping for the agreement levels
agreement_levels = ['Strongly Disagree', 'Disagree', 'Agree', 'Strongly Agree']
data['Opinion'] = pd.Categorical(data['Opinion'], categories=agreement_levels, ordered=True) # categorical data is ordered

# Pivot the data to create a matrix
pivot_data = pd.pivot_table(data, values=None, index='Respondent', columns='Opinion',
                            aggfunc=lambda x: len(x), fill_value=0)

# Create the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_data, cmap='YlGnBu', annot=True, fmt='d', cbar=True)

# Add labels and title
plt.xlabel('Opinion')
plt.ylabel('Respondent')
plt.title('Agreement Heatmap')

# Display the heatmap
plt.show()


# In[6]:


# Define the mapping for the agreement levels
agreement_levels = ['Strongly Disagree', 'Disagree', 'Agree', 'Strongly Agree']
data['Opinion'] = pd.Categorical(data['Opinion'], categories=agreement_levels, ordered=False) ### the categorical data is not ordered



# In[7]:


# Pivot the data to create a matrix
pivot_data = pd.pivot_table(data, values=None, index='Respondent', columns='Opinion',
                            aggfunc=lambda x: len(x), fill_value=0)

# Create the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_data, cmap='YlGnBu', annot=True, fmt='d', cbar=True)

# Add labels and title
plt.xlabel('Opinion')
plt.ylabel('Respondent')
plt.title('Agreement Heatmap')

# Display the heatmap
plt.show()


# In[8]:


# Create the countplot
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='Opinion', palette='YlGnBu')

# Add labels and title
plt.xlabel('Opinion')
plt.ylabel('Count')
plt.title('Distribution of Agreement Levels')

# Rotate x-axis labels if needed
plt.xticks(rotation=45)

# Display the countplot
plt.show()


# In[9]:


# Create the violin plot
plt.figure(figsize=(8, 6))
sns.violinplot(data=data, x='Opinion', y='Respondent', palette='YlGnBu')

# Add labels and title
plt.xlabel('Opinion')
plt.ylabel('Respondent')
plt.title('Distribution of Agreement Levels')

# Display the violin plot
plt.show()


# In[ ]:




