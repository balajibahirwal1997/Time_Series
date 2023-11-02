#!/usr/bin/env python
# coding: utf-8

# In[1]:


from time import time


# In[2]:


# from fbprophet.diagnostics
from prophet import Prophet


# In[3]:


# pip install git+https://github.com/stan-dev/pystan2.git@master


# In[4]:


# from prophet import Cro


# In[5]:


# from prophet import ParameterGrid


# In[6]:


# !pip install darts


# In[ ]:





# In[7]:


# from darts import TimeSeries


# In[8]:


# from darts.models import ExponentialSmoothing


# In[9]:


# from darts.models import Prophet


# In[3]:


import pandas as pd
import numpy as np


# In[4]:


import matplotlib
import time
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os
import plotly.express as px


# In[5]:


# from google.colab import files
# uploaded = files.upload()


# In[6]:


data_1 = pd.read_csv("C:\\Users\\bahirwal\\Desktop\\Balaji\\Analyst\\Time_series\\BRICS_data.csv")


# # New Section

# In[7]:


#import io
#data_1 = pd.read_csv(io.BytesIO(uploaded['BRICS_data.csv']))
data_1.head()


# In[8]:


data_1.shape


# In[9]:


is_range_index = isinstance(data_1.index, pd.RangeIndex)
print(is_range_index)


# In[10]:


data_1['country_name'].value_counts()


# In[11]:


data_1['year'].value_counts()


# In[12]:


time_1 = data_1.copy()


# In[13]:


time_1.dtypes


# In[14]:


time_1['year'] = pd.to_datetime(time_1['year'], format='%Y')


# In[15]:


time_1.tail(20)


# In[16]:


"""# Create a DataFrame with a column containing years from 1960 to 2022
data = {'years': range(1960, 2023)}
df = pd.DataFrame(data)

# Convert the 'years' column to datetime with January 1st as the date
df['date'] = pd.to_datetime(df['years'], format='%Y')
df['date'] = df['date'].dt.replace(month=1, day=1)

# Print the resulting DataFrame
print(df)"""


# In[17]:


# time_1.drop(['Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5','Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8'],axis = 1, inplace = True)


# In[18]:


# Using Boolean mask
mask = time_1['country_name']=='Russian Federation'


# In[19]:


time_2 = time_1[~mask]


# In[20]:


time_2.head()


# In[21]:


is_range_index_1 = isinstance(time_2.index, pd.RangeIndex)
print(is_range_index_1)


# In[22]:


time_2['country_name'].value_counts()


# In[23]:


time_2.dtypes


# In[24]:


time_2.rename(columns = {'year' : 'date'}, inplace = True)
time_2.rename(columns = {'value' : 'y'}, inplace = True)
time_2.rename(columns = {'country_name' : 'country'}, inplace = True)


# In[25]:


time_2['date'] = pd.to_datetime(time_2['date'])
time_2['y'] = time_2['y'].astype(float)


# In[26]:


time_2.head()


# In[27]:


# Creadting datframe for other purpose


# In[28]:


time_data = time_2.copy()


# In[29]:


first_col = time_data.pop('date')


# In[30]:


time_data.insert(0, 'date', first_col)


# In[31]:


time_data.set_index('date', inplace = True)


# In[32]:


time_data.head()


# In[33]:


time_data.country.nunique()


# In[34]:


time_data.tail()


# In[35]:


time_data.dtypes


# In[36]:


time_data.sort_values(['country', 'date']).groupby('country').head()


# In[37]:


type(time_data)


# In[45]:


is_range_index_df1 = isinstance(time_data.index, pd.RangeIndex)
print(is_range_index_df1)


# In[46]:


# dataframe for plotting data


# In[38]:


# Changing dat structure to pivot table solves the problem
time_3 = time_2.pivot(index = 'date', columns = 'country', values = 'y')
time_3.head()


# In[39]:


plt.figure(figsize = (10, 5))
time_3.plot(ax = plt.gca())
plt.xlabel('Year')
plt.ylabel('GDP(10 trillion)')
plt.legend(title = 'country')
plt.show()


# In[40]:


# below graphs show a continuous trend but seasonality is absent(OR hidden)
countries = time_2['country'].unique()

# Create a subplot for each country
fig, axs = plt.subplots(len(countries), figsize=(6, 3 * len(countries)), sharex=True)

# Plot each country's time series in a separate subplot
for i, country in enumerate(countries):
    country_data = time_2[time_2['country'] == country]
    axs[i].plot(country_data['date'], country_data['y'])
    axs[i].set_title(country)
    axs[i].set_xlabel('Date')
    axs[i].set_ylabel('Value of Time Series')

plt.tight_layout()
plt.show()


# In[41]:


"""# Plot timeseries for individual countries
brazil_data = time_2[time_2['country'] == 'Brazil']
plt.figure(figsize = (10, 5))
plt.plot(brazil_data['date'], brazil_data['y'])
plt.xlabel('year')
plt.ylabel('GDP(trillion)')"""


# In[42]:


# FbProphet model
time_4 = time_2.copy()
time_4.head()


# In[43]:


time_4.rename(columns = {'date' : 'ds'}, inplace = True)


# In[44]:


is_range_index_df2 = isinstance(time_4.index, pd.RangeIndex)
print(is_range_index_df2)


# In[45]:


time_4.index


# In[46]:


time_4.index = pd.to_datetime(time_4.index)


# In[47]:


time_4 = time_4.sort_index()


# In[48]:


time_4 = time_4.reset_index()


# In[49]:


time_4['ds'] = pd.to_datetime(time_4['ds'])


# In[50]:


is_range_index_2 = isinstance(time_4.index, pd.RangeIndex)
print(is_range_index_2)


# In[52]:


countries_1 = time_4.groupby('country')


# In[53]:


countries_1.head()


# In[ ]:





# In[54]:


for i in countries_1.groups:
  group = countries_1.get_group(i)
  train = group[(group['ds']  >= '1960-01-01') & ((group['ds']  <= '2019-01-01'))]
  test = group[(group['ds'] > '2019-01-01')]
  print(test.shape)


# In[55]:


group.head()


# In[57]:


from prophet import Prophet
from prophet.diagnostics import cross_validation
from sklearn.model_selection import ParameterGrid


# In[58]:


target = pd.DataFrame()


# In[99]:


for i in countries_1.groups:
    group_1 = countries_1.get_group(i)
    
    m = Prophet(changepoint_prior_scale=0.01, interval_width=0.95, n_changepoints=25)
    m.fit(group_1)

    future = m.make_future_dataframe(periods=3)
    forecast = m.predict(future)

    # Plot the forecast for this group
    fig = m.plot(forecast)

    forecast_1 = forecast.rename(columns={'yhat': 'yhat_' + i})
    #target = target.merge(forecast[['ds', 'yhat_' + i]], on='ds', how='outer')
# arget = pd.merge(target, forecast.set_index('ds'), how = 'outer', left_index = True, right_index = True)


# In[100]:


m.component_modes


# In[101]:


forecast_1.columns


# In[62]:


for i in countries_1.groups:
        group_1 = countries_1.get_group(i)
        m = Prophet()
    # interval_width = 0.95
        m.fit(group_1)
        future = m.make_future_dataframe(periods = 3)
        forecast = m.predict(future)
        m.plot(forecast)
        forecast = forecast.rename(columns = {'yhat' : 'yhat_' + i})
        target = pd.merge(target, forecast.set_index('ds'), how = 'outer', left_index = True, right_index = True)


# In[102]:


target


# In[103]:


target.columns


# In[104]:


target_1 = target[['yhat_Brazil', 'yhat_China', 'yhat_India', 'yhat_South Africa_y']]


# In[105]:


target_1


# In[106]:


target_1.to_csv('target_1.csv', index = True, sep = '~')


# In[109]:


from prophet.diagnostics import cross_validation
df_cv = cross_validation(m, initial='65 days', period='2 days', horizon = '5 days')


# In[ ]:


df_cv.head()


# In[ ]:





# In[ ]:


for i in countries_1.groups:
    group = countries_1.get_group(i)

    # Assuming that 'ds' is the datetime column in your DataFrame
    group['ds'] = pd.to_datetime(group['ds'])  # Convert 'ds' column to datetime if it's not already

    m = Prophet(interval_width=0.95)
    m.fit(group)
    future = m.make_future_dataframe(periods=3)
    forecast = m.predict(future)
    m.plot(forecast)
    forecast = forecast.rename(columns={'yhat': 'yhat_' + i})
    target = pd.merge(target, forecast.set_index('ds'), how='outer', left_index=True, right_index=True)


# In[ ]:


# Hyper Parameter tuning


# In[ ]:


import pandas as pd
from fbprophet import Prophet
from sklearn.model_selection import GridSearchCV
from fbprophet.diagnostics import cross_validation

# Load your time series data into a DataFrame
# Assuming your data has 'ds' and 'y' columns representing dates and target values

# Create a parameter grid for hyperparameter tuning
param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5, 1.0],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
}

# Initialize the Prophet model
model = Prophet()

# Create a GridSearchCV object
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the model to your data while performing hyperparameter tuning
grid_search.fit(your_data_frame)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Get the best model with the optimal hyperparameters
best_model = grid_search.best_estimator_

# Perform cross-validation to evaluate the model
df_cv = cross_validation(best_model, initial='365 days', period='180 days', horizon='365 days')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


## Best model with HyperParameter Tunned


# In[ ]:


countries_2 = time_4.groupby('country')
countries_2.head()


# In[ ]:


print(countries_2)


# In[ ]:


for i in countries_2.groups:
  group_1 = countries_2.get_group(i)
  train_1 = group_1[(group_1['ds']  >= '1960-01-01') & ((group_1['ds']  <= '2019-01-01'))]
  test_1 = group_1[(group_1['ds'] > '2019-01-01')]
  print(test_1.shape)


# In[ ]:


target_1 = pd.DataFrame()


# In[ ]:




