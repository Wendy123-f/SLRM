#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


#Reading in our data
data = pd.read_excel("C:\\Users\\Faith\\Desktop\\Faith Docs\\Book1.xlsx")
data.head(5)


# In[3]:


column_names = {'X1':'Relative_Compactness',
                'X2': 'Surface_Area', 
                'X3':  'Wall_Area', 'X4': 'Roof_Area',
                'X5': 'Overall_Height',
                'X6': 'Orientation',
                'X7': 'Glazing_Area', 
                'X8': 'Glazing_Area_Distribution', 
                'Y1': 'Heating_Load',
                'Y2': 'Cooling_Load'}

data = data.rename(columns = column_names)

#Select a sample of the dataset
slr = data[['Relative_Compactness', 'Cooling_Load']].sample(15, random_state=2)
sns.regplot(x="Relative_Compactness", y="Cooling_Load", data=slr)


# In[4]:


#We normalize our dataset to a common scale using the min max scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalised_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
X = normalised_data.drop(columns=['Heating_Load', 'Cooling_Load'])
y = normalised_data['Heating_Load']


# In[5]:


#Now, we split our dataset into the training and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[6]:


from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()


# In[7]:


#fit the model to the training dataset
linear_model.fit(X_train, y_train)

#obtain predictions
predicted_values = linear_model.predict(X_test)


# In[8]:


#Mean absolute Error(MAE)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, predicted_values)
round(mae, 3)


# In[9]:


from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, predicted_values))
round(rmse, 3)


# In[10]:


from sklearn.metrics import r2_score
r2_score = r2_score(y_test, predicted_values)
round(r2_score, 3)


# In[11]:


from sklearn.linear_model import Ridge
ridge_reg =  Ridge(alpha = 0.5)
ridge_reg.fit(X_train, y_train)


# In[21]:


from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha = 0.001)
lasso_reg.fit(X_train, y_train)

#Comparing the effects of regularisation
def get_weights_data(model, feat, col_name):#This function returns the weight of every feature
    weights = pd.Series(model.coef_, feat.columns).sort_values()
    weights_data = pd.DataFrame(weights).reset_index()
    weights_data.columns = ['features', col_name]
    weights_data[col_name].round(3)
    return weights_data


# In[27]:


linear_model_weights = get_weights_data(linear_model, X_train, 'Linear_Model_Weight')
ridge_weights_data = get_weights_data(ridge_reg, X_train, 'Ridge_Weight')
lasso_weights_data = get_weights_data(lasso_reg, X_train, 'Lasso_weight')
final_weights = pd.merge(linear_model_weights, ridge_weights_data, on='features')
final_weights = pd.merge(final_weights, lasso_weights_data, on='features')


# In[29]:


print(final_weights)


# In[ ]:




