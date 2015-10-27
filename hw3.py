# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 2015

@author: violetgirl
"""
import pandas as pd
import numpy as np
import seaborn as sb
import scipy
import matplotlib.pyplot as plt

# load gapminder dataset
data = pd.read_csv('gapminder.csv',low_memory=False)
# lower-case all DataFrame column names
data.columns = map(str.lower, data.columns)
# bug fix for display formats to avoid run time errors
pd.set_option('display.float_format', lambda x:'%f'%x)

# setting variables to be numeric
data['suicideper100th'] = data['suicideper100th'].convert_objects(convert_numeric=True)
data['breastcancerper100th'] = data['breastcancerper100th'].convert_objects(convert_numeric=True)
data['hivrate'] = data['hivrate'].convert_objects(convert_numeric=True)
data['employrate'] = data['employrate'].convert_objects(convert_numeric=True)

# display summary statistics about the data
# print("Statistics for a Suicide Rate")
# print(data['suicideper100th'].describe())

# subset data for a high suicide rate based on summary statistics
sub = data[(data['suicideper100th']>12)]
# make a copy of my new subsetted data
sub_copy = sub.copy()
# remove missing values
sub_copy=sub_copy.dropna()

# Bivariate graph for association of breast cancer rate with HIV rate for people with a high suicide rate
plt.figure(1)
sb.regplot(x="employrate",y="breastcancerper100th",fit_reg=True,data=sub_copy)
plt.xlabel('Employment Rate')
plt.ylabel('Breast Cancer Rate')
plt.title('Breast Cancer Rate vs. Employment Rate for People with a High Suicide Rate')

plt.figure(2)
sb.regplot(x="employrate",y="hivrate",fit_reg=True,data=sub_copy)
plt.xlabel('Employment Rate')
plt.ylabel('HIV Rate')
plt.title('HIV Rate vs. Employment Rate for People with a High Suicide Rate')

print ('association between breast cancer rate and employment rate')
print('r coefficient and p value')
print (scipy.stats.pearsonr(sub_copy['breastcancerper100th'],sub_copy['employrate']))
# r^2 = 0.1743 => 17.43%

print ('association between HIV rate and employment rate')
print('r coefficient and p value')
print (scipy.stats.pearsonr(sub_copy['hivrate'],sub_copy['employrate']))
# r^2 = 0.006%

# END