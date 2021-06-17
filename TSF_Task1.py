#!/usr/bin/env python
# coding: utf-8

# In[59]:


#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[60]:


#getting the dataset ready
dataset = 'http://bit.ly/w-data'
data = pd.read_csv(dataset)
print("Good Job! The first 10 data are shown below")
data.head(10)


# In[61]:


#plotting scatter plot
data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours Vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# In[62]:


#dataset description
data.describe()


# In[63]:


#dataset (rows, columns)
data.shape


# In[64]:


#Model training Stage
x = data.iloc[:,:-1].values
y = data.iloc[:, 1].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)
print("Model Successfully Trained")


# In[65]:


#Getting the regression line
line = regressor.coef_*x+regressor.intercept_

plt.scatter(x,y)
plt.plot(x, line, color = 'purple')
plt.text(6,30,("Intercept = {}".format(round(regressor.intercept_))))
plt.text(6,23,("Slope = {}".format(regressor.coef_)))
plt.show()


# In[66]:


#Prediction Stage
print(x_test)
y_predict = regressor.predict(x_test)
print(y_predict)


# In[67]:


#Obtaining Actual/Predicted Values
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})
df


# In[68]:


#Performance of my model
print("Mean Absolute Error: ",metrics.mean_absolute_error(y_test,y_predict))
print("Mean Squared Error: ",metrics.mean_squared_error(y_test,y_predict))
print("RMSE: ", np.sqrt(metrics.mean_absolute_error(y_test,y_predict)))
print("Maximum Error: ", metrics.max_error(y_test,y_predict))

