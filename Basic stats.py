#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings

# To ignore all warnings
warnings.filterwarnings("ignore")


# # Step 1: Data Loading and Exploration

# In[2]:


import pandas as pd


# In[3]:


# Load the dataset
data = pd.read_csv('housing_prices.csv')


# In[4]:


# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)


# In[5]:


# Explore the datas
df.head()  # Display the first few rows


# In[6]:


df.describe()  # Summary statistics


# In[7]:


df.info() # Information about columns and data types


# In[8]:


# Explore missing values
df.isnull().sum()


# # Step 2: Data Preprocessing

# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[10]:


# Handling missing values
df.dropna(inplace=True)


# In[11]:


# Drop duplicates if any
df.drop_duplicates(inplace=True)


# In[12]:


# Check for outliers using a boxplot
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['Price'])
plt.title('Boxplot of Housing Prices')
plt.xlabel('Price')
plt.show()


# # Step 3: Statistical Analysis

# In[13]:


from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
import seaborn as sns


# In[14]:


# 1. Mean, Median, and Mode
mean_price = df['Price'].mean()
median_price = df['Price'].median()
mode_price = df['Price'].mode()


# In[15]:


# 2. Variance and Standard Deviation
variance_price = df['Price'].var()
std_dev_price = df['Price'].std()


# In[16]:


# 3. Correlation
correlation = df.corr()


# In[17]:


# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[18]:


# 4. Linear Regression
X = df[['Area']]  # Independent variable (predictor)
y = df['Price']   # Dependent variable (target)
model = LinearRegression()
model.fit(X, y)


# In[19]:


# Step 5: Hypothesis Testing (T-test) - Compare 'MaintenanceStaff' and 'SwimmingPool'
maintenance_staff = data[data['MaintenanceStaff'] == 1]['Price']
no_maintenance_staff = data[data['MaintenanceStaff'] == 0]['Price']

t_stat, p_value = ttest_ind(maintenance_staff, no_maintenance_staff)
print("T-test results:")
print("T-statistic:", t_stat)
print("P-value:", p_value)


# In[20]:


#6. T-test
neighborhood_A_prices = df[df['Location'] == 'Neighborhood_A']['Price']
neighborhood_B_prices = df[df['Location'] == 'Neighborhood_B']['Price']

t_stat, p_value = stats.ttest_ind(neighborhood_A_prices, neighborhood_B_prices)

if p_value < 0.05:
    print("There's a significant difference in average prices between Neighborhood_A and Neighborhood_B.")
else:
    print("There's no significant difference in average prices between Neighborhood_A and Neighborhood_B.")


# In[21]:


#7.ANOVA
neighborhoods = df['Location'].unique()
neighborhood_groups = [df[df['Location'] == neighborhood]['Price'] for neighborhood in neighborhoods]

f_statistic, p_value = f_oneway(*neighborhood_groups)

if p_value < 0.05:
    print("There's a significant difference in average prices among different neighborhoods.")
else:
    print("There's no significant difference in average prices among different neighborhoods.")


# # Step 4: Data Visualization

# In[22]:


import plotly.express as px


# In[23]:


fig = px.scatter(data, x='Area', y='Price', trendline='ols')
fig.show()


# In[24]:


# Regression plot for 'Area' vs. 'Price'
plt.figure(figsize=(8, 6))
sns.regplot(x='Area', y='Price', data=data)
plt.title('Regression Plot: Price vs Area')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()


# In[25]:


# Example: Visualizing 'Area', 'Price', and 'No. of Bedrooms'
fig = px.scatter_3d(data, x='Area', y='Price', z='No. of Bedrooms', color='Price')
fig.update_layout(scene=dict(xaxis_title='Area', yaxis_title='Price', zaxis_title='No. of Bedrooms'))
fig.show()


# In[26]:


fig_3d = px.scatter_3d(data, x='Area', y='No. of Bedrooms', z='Price', color='Price')
fig_3d.update_traces(marker=dict(size=3))
fig_3d.show()


# In[ ]:




