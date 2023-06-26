#!/usr/bin/env python
# coding: utf-8

# In[3]:


### coefficients
import numpy as np
import pandas as pd


# In[4]:


# Set random seed for reproducibility
np.random.seed(0)

# Gene expression levels (0-100)
gene_expression = np.random.randint(0, 101, 10)

# Obesity levels (1-5)
obesity = np.random.randint(1, 6, 10)

# Create a DataFrame with gene expression and obesity levels
data = pd.DataFrame({'Gene Expression': gene_expression, 'Obesity Level': obesity})

# Display the dataset
print(data)


# In[ ]:


"""
We can assume the gene expression levels range from 0 to 100, 
and the obesity levels are measured on a scale from 1 to 5. 
Above is an example dataset for 10 individuals:
"""


# In[ ]:


"""
To test the coefficient of the relationship between gene expression and obesity, 
you can use linear regression. 
"""


# In[5]:


import statsmodels.api as sm

# Define the predictor variable (gene expression) and the response variable (obesity)
X = data['Gene Expression']
y = data['Obesity Level']

# Add a constant term to the predictor variable
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X)
results = model.fit()

# Print the coefficient of the gene expression variable
print('Coefficient:', results.params['Gene Expression'])


# In[ ]:


"""
we first define the predictor variable (X) as the gene expression and the response variable (y) as the obesity level. 
We then add a constant term to the predictor variable using sm.add_constant().

Next, we fit the linear regression model using sm.OLS(), 
passing the response variable (y) and the predictor variable with the constant (X). 
The fit() method is used to obtain the regression results.

Finally, we print the coefficient of the gene expression variable using results.params['Gene Expression']. 
This coefficient represents the estimated effect of gene expression on obesity.

Remember to import the necessary libraries (numpy, pandas, and statsmodels.api) 
at the beginning of your code to run the examples successfully.
"""

