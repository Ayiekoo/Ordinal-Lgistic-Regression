#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


# Set random seed for reproducibility
np.random.seed(0)

# Generate hypothetical data with Likert scale ratings
data = pd.DataFrame({
    'Customer': range(1, 11),
    'Rating': np.random.randint(1, 6, 10), # Likert scale range
    'Product': np.random.choice(['A', 'B', 'C'], 10)
})

# Display the dataset
print(data) ## prints 10 customers with ratings from 1 to 5


# In[ ]:


"""
In this code, we assume that the Likert scale ranges from 1 to 5, 
but you can adjust the range to match your specific Likert scale, 
such as 1 to 7 or any other range you have used.

The rest of the code, including the multinomial logistic regression part, 
can remain the same. 
It will handle the Likert scale data as ordinal variables and perform the regression analysis accordingly.
"""


# In[3]:


import statsmodels.api as sm


# In[7]:


# Define the predictor variables (product) and the response variable (rating)
X = pd.get_dummies(data['Product'], drop_first=True)
y = data['Rating']

# Add a constant term to the predictor variables
X = sm.add_constant(X)

# Fit the multinomial logistic regression model
model = sm.MNLogit(y, X)
results = model.fit()

# Print the summary of the model
print(results.summary())

"""
This prints the dependent variable, rating, coefficient, standard error
"""


# In[ ]:


"""
In this code, we first define the predictor variables (X) as the product using one-hot encoding. 
We use pd.get_dummies() to create dummy variables for the product categories (A, B, and C), 
and set drop_first=True to drop the first category to avoid multicollinearity.


Next, we define the response variable (y) as the rating. 
We then add a constant term to the predictor variables using sm.add_constant().


We fit the multinomial logistic regression model using sm.MNLogit(), 
passing the response variable (y) and the predictor variables with the constant (X). 
The fit() method is used to obtain the regression results.


Finally, we print the summary of the model using results.summary(). 
This will provide information about the estimated coefficients, standard errors, p-values, and other model statistics.


Remember to import the necessary libraries (numpy, pandas, and statsmodels.api) at the beginning of your code to run the examples successfully.
"""


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[12]:


# Hypothetical data
data = pd.DataFrame({
    'Customer': range(1, 11),
    'Rating': [3, 2, 4, 5, 1, 3, 4, 2, 5, 3],
    'Product': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A']
})

# Pivot the data to create a matrix
pivot_data = data.pivot_table(index='Customer', columns='Product', values='Rating')

# Create the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_data, cmap='YlGnBu', annot=True, fmt='.2f', cbar=True)

# Add labels and title
plt.xlabel('Product')
plt.ylabel('Customer')
plt.title('Rating Heatmap')

# Display the heatmap
plt.show()


# In[ ]:




