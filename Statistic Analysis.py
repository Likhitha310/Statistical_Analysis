#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings

# To ignore all warnings
warnings.filterwarnings("ignore")


# # Descriptive Statistics

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


# Load the census data from a CSV file
data = pd.read_csv('census.csv')


# In[4]:


print(data.head())  # Display the first few rows of the dataset


# In[5]:


print(data.info())  # Show information about the dataset


# In[6]:


# Display basic statistics using NumPy
districts = data['District']
states = data['State']
area = data['Area_km2']
population = data['Population']
growth = data['Growth']
sex_ratio = data['Sex_Ratio']
literacy = data['Literacy']


# In[7]:


population = [1, 2, 3, 4, 5]


# In[8]:


# Calculate basic statistics using NumPy
population_mean = np.mean(population)
population_std = np.std(population)
population_max = np.max(population)
population_min = np.min(population)


# In[9]:


# Display basic statistics
print(f"Population Mean: {population_mean}")
print(f"Population Standard Deviation: {population_std}")
print(f"Maximum Population: {population_max}")
print(f"Minimum Population: {population_min}")


# In[10]:


# Visualization 1: Population Distribution
plt.figure(figsize=(4, 2))
plt.hist(population, bins=10, color='skyblue', edgecolor='black')
plt.title('Population Distribution')
plt.xlabel('Population')
plt.ylabel('Frequency')
plt.show()


# In[11]:


# Visualization 2: Growth Rate
plt.figure(figsize=(8, 6))
plt.plot(districts, growth, marker='o', linestyle='-')
plt.title('Growth Rate by District')
plt.xlabel('District')
plt.ylabel('Growth Rate')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[12]:


# Convert numeric states to strings
states = [str(state) for state in states]


# In[13]:


# Visualization 3: Literacy Rate Distribution
plt.figure(figsize=(6, 4))
plt.bar(states, literacy, color='orange')
plt.title('Literacy Rate by State')
plt.xlabel('State')
plt.ylabel('Literacy Rate')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# # Inferential statistics

# In[14]:


import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy import stats


# In[15]:


# Load the census data from a CSV file
data = pd.read_csv('students.csv')


# In[16]:


# Sample data as a dictionary

data = {
    'Gender': ['female', 'female', 'female', 'male'],
    'EthnicGroup': ['', 'group C', 'group B', 'group A'],
    'ParentEduc': ["bachelor's degree", 'some college', "master's degree", "associate's degree"],
    'LunchType': ['standard', 'standard', 'standard', 'free/reduced'],
    'TestPrep': ['none', '', 'none', 'none'],
    'ParentMaritalStatus': ['married', 'married', 'single', 'married'],
    'PracticeSport': ['yes', 'yes', 'yes', 'no'],
    'IsFirstChild': [3, 0, 4, 1],
    'NrSiblings': [None, None, None, None],
    'TransportMeans': ['school_bus', None, 'school_bus', None],
    'WklyStudyHours': ['< 5', '05-Oct', '< 5', '05-Oct'],
    'MathScore': [71, 69, 87, 45],
    'ReadingScore': [71, 90, 93, 56],
    'WritingScore': [74, 88, 91, 42]
}



# In[17]:


# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)


# In[18]:


# Handling empty string values in the dataframe
df = df.replace('', pd.NA)


# In[19]:


# Handling empty numeric values (e.g., NrSiblings)
df['NrSiblings'] = df['NrSiblings'].fillna(0)


# In[20]:


# Calculate means
mean_test_prep = test_prep_math_scores.mean()
mean_no_test_prep = no_test_prep_math_scores.mean()


# In[ ]:


# T-test to check the effect of TestPrep on Math scores
test_prep_math_scores = df[df['TestPrep'] != 'none']['MathScore']
no_test_prep_math_scores = df[df['TestPrep'] == 'none']['MathScore']

t_stat, p_value = stats.ttest_ind(no_test_prep_math_scores, test_prep_math_scores)


# In[ ]:


# Print the results of the t-test
alpha = 0.05
if p_value < alpha:
    print("Test Preparation significantly affects Math scores.")
else:
    print("Test Preparation does not significantly affect Math scores.")


# In[ ]:


# Create a bar plot to visualize the means
plt.figure(figsize=(4, 4))
plt.bar(['With Test Prep', 'No Test Prep'], [mean_test_prep, mean_no_test_prep], color=['black', 'grey'])
plt.title('Mean Math Scores with and without Test Preparation')
plt.xlabel('Test Preparation')
plt.ylabel('Mean Math Score')
plt.show()


# # Regression Analysis

# In[21]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[22]:


data = pd.read_csv('housing_prices.csv')


# In[23]:


print(data.head())


# In[24]:


# Handle missing values
data.dropna(inplace=True)


# In[25]:


# Check for missing values and handle them if necessary
print(data.isnull().sum())


# In[26]:


# Display the column names in your dataset
print(data.columns)


# In[27]:


# Pairplot to visualize relationships between numerical features
sns.pairplot(data, vars=['Area', 'No. of Bedrooms', 'Price'])
plt.show()


# In[28]:


# Boxplot for categorical variables against price
sns.boxplot(x='Location', y='Price', data=data)
plt.xticks(rotation=90)
plt.show()


# In[29]:


# Define features (X) and target variable (y)
X = data[['Area', 'No. of Bedrooms']]
y = data['Price']


# In[30]:


# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[31]:


# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# In[32]:


# Predict on the test set
y_pred = model.predict(X_test)


# In[33]:


y_pred


# # Time Series Analysis

# In[34]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[35]:


# Load the data
data = pd.read_csv('stock_price.csv')  # Replace 'your_file.csv' with the actual file path


# In[36]:


data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' to a datetime object
data.set_index('Date', inplace=True)  # Set 'Date' as the index


# In[37]:


# Visualize the stock prices
plt.figure(figsize=(6, 4))
plt.plot(data['Close'], label='Close Price')
plt.title('Historical Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[38]:


# Correlation Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[39]:


plt.figure(figsize=(12, 6))
sns.violinplot(x='Close', data=data, color='brown')
plt.title('Distribution of Close Prices')
plt.xlabel('Close Price')
plt.show()


# # Survival Analysis

# In[40]:


import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt


# In[41]:


data = pd.read_csv('survival.csv')


# In[42]:


# Fit Kaplan-Meier estimator
kmf = KaplanMeierFitter()
kmf.fit(data['Time'], event_observed=data['Event'])


# In[43]:


# Kaplan-Meier Plot
plt.figure(figsize=(10, 4))
kmf.plot()
plt.title('Kaplan-Meier Survival Curve')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.show()


# # Factor Analysis

# In[44]:


import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[45]:


# Load the dataset
data = pd.read_csv('factor.csv')  # Replace 'your_dataset.csv' with the file path or URL


# In[46]:


# Handle categorical variables (convert to dummy variables)
data = pd.get_dummies(data)


# In[47]:


# Handle missing values
data.fillna(data.mean(), inplace=True)  # Replace missing values with the mean


# In[48]:


# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)


# In[49]:


# Fit Factor Analysis
n_factors = 5  # Define the number of factors you want
fa = FactorAnalysis(n_components=n_factors)
fa.fit(data_scaled)


# In[50]:


# Get the loadings (factor weights)
loadings = fa.components_
loadings_df = pd.DataFrame(loadings, columns=data.columns)


# In[51]:


# Visualizing factor loadings
plt.figure(figsize=(10, 4))
plt.imshow(loadings_df, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.title('Factor Loadings')
plt.xlabel('Variables')
plt.ylabel('Factors')
plt.show()


# # Cluster Analysis

# In[52]:


import pandas as pd
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# In[53]:


# Load the dataset
data = pd.read_csv('factor.csv')  # Replace 'your_dataset.csv' with the file path or URL


# In[54]:


df = pd.DataFrame(data)


# In[55]:


# Prepare data for clustering
X = df[['Age', 'Purchase Amount (USD)']]


# In[56]:


# Implement KMeans clustering
kmeans = KMeans(n_clusters=3)  # Changing the number of clusters
kmeans.fit(X)
df['Cluster'] = kmeans.labels_


# In[57]:


# Visualizing the clusters
plt.scatter(X['Age'], X['Purchase Amount (USD)'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Purchase Amount (USD)')
plt.title('Cluster Analysis based on Age and Purchase Amount')
plt.show()


# # ANOVA

# In[58]:


import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[59]:


# Load the dataset
data = pd.read_csv('factor.csv')  # Replace 'your_dataset.csv' with the file path or URL


# In[60]:


df = pd.DataFrame(data)


# In[61]:


model = ols('Q("Purchase Amount (USD)") ~ C(Category)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)


# In[62]:


# Print ANOVA results
print(anova_table)


# In[63]:


# Example violin plot using Seaborn
plt.figure(figsize=(8, 6))
sns.violinplot(x='Category', y='Purchase Amount (USD)', data=df)
plt.title('Distribution of Purchase Amount by Category')
plt.xlabel('Category')
plt.ylabel('Purchase Amount (USD)')
plt.show()


# # PCA

# In[64]:


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[65]:


data = pd.read_csv('pca.csv')  # Replace 'your_dataset.csv' with the file path or URL


# In[66]:


# Inspect column names
print(data.columns)


# In[67]:


# Display the data types in each column to identify non-numeric data
print(data.dtypes)


# In[68]:


# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)


# In[69]:


# Apply PCA
pca = PCA(n_components=2)  # Choose the number of components (e.g., 2)
pca_result = pca.fit_transform(scaled_data)


# In[70]:


# You can create a new DataFrame to store the PCA results for visualization
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])


# In[71]:


# The amount of variance that each PC explains
explained_variance = pca.explained_variance_ratio_
print("Explained variance by PC1 and PC2:", explained_variance)


# In[72]:


# Visualizing the PCA results
# You can create a scatter plot to visualize the data points in the PCA space
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
plt.title('PCA of Your Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.show()


# # Multivariate Analysis

# In[73]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[74]:


# Load the dataset
data = pd.read_csv('students.csv')


# In[75]:


# Correlation matrix
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix')
plt.show()



# In[76]:


# Boxplot comparison of test scores among different 'ParentEduc' levels
plt.figure(figsize=(10, 6))
sns.boxplot(x='ParentEduc', y='MathScore', data=data)
plt.title('Math Scores across Parental Education Levels')
plt.xticks(rotation=45)
plt.show()


# In[77]:


# Categorical variable analysis - Count plot of 'TestPrep'
plt.figure(figsize=(8, 6))
sns.countplot(x='TestPrep', data=data, hue='PracticeSport')
plt.title('Test Preparation count by Sports Practice')
plt.show()


# In[ ]:





# In[ ]:




