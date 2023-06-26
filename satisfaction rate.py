#!/usr/bin/env python
# coding: utf-8

# In[1]:


## sample data

satisfaction_data = [3, 2, 4, 5, 1, 3, 4, 2, 5, 3]


# In[2]:


### let's create a bar plot
import matplotlib.pyplot as plt

def draw_tower_cut(data):
    labels = ['1', '2', '3', '4', '5']
    counts = [data.count(1), data.count(2), data.count(3), data.count(4), data.count(5)]
    
    plt.bar(labels, counts)
    plt.xlabel('Satisfaction Level')
    plt.ylabel('Count')
    plt.title('Tower Cut of Satisfaction Levels')
    plt.show()

# Example usage
draw_tower_cut(satisfaction_data)


# In[3]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


# let's draw a heat map
satisfaction_data = [
    [3, 2, 4, 5, 1],
    [3, 4, 2, 5, 3],
    [4, 2, 5, 3, 1],
    [2, 5, 3, 1, 4],
    [5, 3, 4, 2, 1]
]

# Create a numpy array from the data
satisfaction_array = np.array(satisfaction_data)

# Create the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(satisfaction_array, cmap='YlGnBu', annot=True, fmt='d', cbar=True)

# Add labels and title
plt.xlabel('Ordinal Response')
plt.ylabel('Individuals')
plt.title('Satisfaction Levels Heatmap')

# Display the heatmap
plt.show()


# In[ ]:




