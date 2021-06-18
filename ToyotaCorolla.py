# -*- coding: utf-8 -*-
"""
Created on Sun May  9 13:45:18 2021

@author: DELL
"""

#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset
Toyota = pd.read_csv('ToyotaCorolla.csv',encoding='ISO-8859-1')
Toyota1= Toyota.iloc[:,[2,3,6,8,12,13,15,16,17]]
Toyota1.rename(columns={'Age_08_04':'Age'},inplace= True)

#EDA
eda=Toyota1.describe()

#Visualization
plt.boxplot(Toyota1['Price'])
plt.boxplot(Toyota1['Age'])
plt.boxplot(Toyota1['KM'])
plt.boxplot(Toyota1['HP'])
plt.boxplot(Toyota['cc'])
plt.boxplot(Toyota['Quarterly_Tax'])
plt.boxplot(Toyota['Weight'])
#All the data are not normally distributed. Price,Age,KM,HP,cc,Quarterly_Tax,Weight is having too many outliers

import statsmodels.api as sm
sm.graphics.qqplot(Toyota1['Price'])#shows the data 'Price' is not normal
sm.graphics.qqplot(Toyota1['Age'])#shows the data 'Age' is not normal, Age is a descrete count
sm.graphics.qqplot(Toyota1['KM'])#'KM' is not normal
sm.graphics.qqplot(Toyota1['HP'])# Data is a descrete count
sm.graphics.qqplot(Toyota1['cc'])#Data is a descrete count
sm.graphics.qqplot(Toyota1['Doors'])#Data is a descrete categorical
sm.graphics.qqplot(Toyota1['Gears'])#Data is a descrete categorical
sm.graphics.qqplot(Toyota1['Quarterly_Tax'])#data is a descrete count and 'Quarterly_Tax' is not normal
sm.graphics.qqplot(Toyota['Weight'])#Data is not normal and it shows discreate count

plt.hist(Toyota1['Price'])#Price is Right skewed
plt.hist(Toyota1['Age'])# Age is left skewed
plt.hist(Toyota1['KM'])# KM is Right skewed
plt.hist(Toyota1['HP'])# Data is very unevenly distributed and left skewed
plt.hist(Toyota1['Quarterly_Tax'])#Data is unevenly distributed and right skewed
plt.hist(Toyota1['Weight'])#Weight is right skewed
#Doors and Gears are categorical data and is repeated

import seaborn as sns
sns.pairplot(Toyota1)

#Correlation
Correlation_values= Toyota1.corr()

#Building the model
import statsmodels.formula.api as smf
ml1 = smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data=Toyota1).fit()
ml1.summary()
#Here R-squared=0.86,Adj. R-Squared=0.86,
#cc and Doors are insignificant(P value cc=0.179,Doors=0.968)

#Building an individual model
ml1 = smf.ols("Price~cc",data=Toyota1).fit()
ml1.summary()
#cc is significant when built alone(p value cc = 0.00)

ml1 = smf.ols("Price~Doors",data=Toyota1).fit()
ml1.summary()
#Doors is significant when built alone(p value Doors = 0.00)

ml1_together = smf.ols('Price~cc+Doors',data=Toyota1).fit()
ml1_together.summary()
#Both cc and Doors are significant (p value is 0.00 for both)

#Plotting the influential plot
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)

#Removing 80 and checking for the significance
Toyota2 = Toyota1.drop(Toyota.index[[80]],axis=0)
ml2 = smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data=Toyota2).fit()
ml2.summary()
#cc is significant and Doors is insignificant (0.48)

#Removing 80,221 (as 221 is the next influential pt.)
Toyota3 = Toyota1.drop(Toyota.index[[80,221]],axis=0)
ml3 = smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data=Toyota3).fit()
ml3.summary()
#Doors is insignificant (p value is 0.087)

#Removing 80,221,960(as 960 is the next influential pt.after 80 and 221)
Toyota4 = Toyota1.drop(Toyota.index[[80,221,960]],axis=0)
ml4 = smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data=Toyota4).fit()
ml4.summary()
#Finally all the variables are significant and hence, we will select this as the final model.
Final_model = smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data=Toyota4).fit()
Final_model.summary()
#Here, R- squared=0.885, Adj. R-squared=0.885,P values significant for all the variables.

#Prediction
Final_model_pred= Final_model.predict(Toyota4)

#Visualization using scatter plot between Price and Predicted Price
plt.scatter(Toyota4["Price"],Final_model_pred,color='red');plt.xlabel("PRICE");plt.ylabel("PREDICTED PRICE")
#The Actual Price and the predicted price are linear.

#Visualization of predicted price and the residuals
plt.scatter(Final_model_pred,Final_model.resid_pearson, color="green");plt.xlabel("Predicted Price");plt.ylabel("Residuals")

#RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(Toyota4.Price,Final_model_pred))
rmse
#1227.47 Euros
