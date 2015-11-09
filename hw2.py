# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 2015

@author: violetgirl
"""
import pandas as pd
import numpy as np
import scipy.stats 
import seaborn as sb
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
#print("Statistics for a Suicide Rate")
#print(data['suicideper100th'].describe())

# subset data for a high suicide rate based on summary statistics
sub = data[(data['suicideper100th']>12)]
#make a copy of my new subsetted data
sub_copy = sub.copy()
# remove missing values 
sub_copy=sub_copy.dropna()

# EMPLOYMENT RATE
# group the data in 2 groups and record it into new variable ecgroup4
def ecgroup4 (row):
    if row['employrate'] >= 32 and row['employrate'] < 45: 
        return 1 
    elif row['employrate'] >= 45 and row['employrate'] < 58: 
        return 2 
    elif row['employrate'] >= 58 and row['employrate'] < 71: 
        return 3 
    elif row['employrate'] >= 71 and row['employrate'] < 84:
        return 4 
        
sub_copy['ecgroup4'] = sub_copy.apply(lambda row:  ecgroup4 (row), axis=1)

def hcgroup4 (row):
    if row['hivrate'] >= 0 and row['hivrate'] < 1 :
        return 1
    elif row['hivrate'] >= 1 and row['hivrate'] < 26:
        return 2
    #elif row['hivrate'] >= 7  and row['hivrate'] < 26:
     #   return 3
sub_copy['hcgroup4'] = sub_copy.apply(lambda row:  hcgroup4 (row), axis=1)

# contingency table of observed counts
ct1=pd.crosstab(sub_copy['hcgroup4'], sub_copy['ecgroup4'])
print (ct1)

# column percentages
colsum=ct1.sum(axis=0)
colpct=ct1/colsum
print(colpct)

# chi-square
print ('chi-square value, p value, expected counts')
cs1= scipy.stats.chi2_contingency(ct1)
print (cs1)

# plot results to understand a relationship
# set explanatory variable to categoric
sub_copy['ecgroup4']=sub_copy['ecgroup4'].astype('category')
# set response variable to numeric
sub_copy['hcgroup4']=sub_copy['hcgroup4'].convert_objects(convert_numeric=True)

sb.factorplot(x='ecgroup4',y='hcgroup4',data=sub_copy,kind="bar",ci=None)
plt.xlabel("Employment rate")
plt.ylabel("HIV rate")

# Post-hoc test
# Chi test for the first pair 1-2
recode2 = {1: 1, 2: 2}
sub_copy['comp_var_12']= sub_copy['ecgroup4'].map(recode2)
# contingency table of observed counts
ct1=pd.crosstab(sub_copy['hcgroup4'], sub_copy['comp_var_12'])
print (ct1)
# column percentages
colsum=ct1.sum(axis=0)
colpct=ct1/colsum
print(colpct)
# chi-square
print ('chi-square value for 1-2 pair, p value, expected counts')
cs1= scipy.stats.chi2_contingency(ct1)
print (cs1)

# Chi test for the first pair 1-3
recode2 = {1: 1, 3: 3}
sub_copy['comp_var_13']= sub_copy['ecgroup4'].map(recode2)
# contingency table of observed counts
ct1=pd.crosstab(sub_copy['hcgroup4'], sub_copy['comp_var_13'])
print (ct1)
# column percentages
colsum=ct1.sum(axis=0)
colpct=ct1/colsum
print(colpct)
# chi-square
print ('chi-square value for 1-3 pair, p value, expected counts')
cs1= scipy.stats.chi2_contingency(ct1)
print (cs1)

# Chi test for the first pair 1-4
recode2 = {1: 1, 4: 4}
sub_copy['comp_var_14']= sub_copy['ecgroup4'].map(recode2)
# contingency table of observed counts
ct1=pd.crosstab(sub_copy['hcgroup4'], sub_copy['comp_var_14'])
print (ct1)
# column percentages
colsum=ct1.sum(axis=0)
colpct=ct1/colsum
print(colpct)
# chi-square
print ('chi-square value for 1-4 pair, p value, expected counts')
cs1= scipy.stats.chi2_contingency(ct1)
print (cs1)

# Chi test for the first pair 2-3
recode2 = {2: 2, 3: 3}
sub_copy['comp_var_23']= sub_copy['ecgroup4'].map(recode2)
# contingency table of observed counts
ct1=pd.crosstab(sub_copy['hcgroup4'], sub_copy['comp_var_23'])
print (ct1)
# column percentages
colsum=ct1.sum(axis=0)
colpct=ct1/colsum
print(colpct)
# chi-square
print ('chi-square value for 2-3 pair, p value, expected counts')
cs1= scipy.stats.chi2_contingency(ct1)
print (cs1)

# Chi test for the first pair 2-4
recode2 = {2: 2, 4: 4}
sub_copy['comp_var_24']= sub_copy['ecgroup4'].map(recode2)
# contingency table of observed counts
ct1=pd.crosstab(sub_copy['hcgroup4'], sub_copy['comp_var_24'])
print (ct1)
# column percentages
colsum=ct1.sum(axis=0)
colpct=ct1/colsum
print(colpct)
# chi-square
print ('chi-square value for 2-4 pair, p value, expected counts')
cs1= scipy.stats.chi2_contingency(ct1)
print (cs1)

# Chi test for the first pair 3-4
recode2 = {3: 3, 4: 4}
sub_copy['comp_var_34']= sub_copy['ecgroup4'].map(recode2)
# contingency table of observed counts
ct1=pd.crosstab(sub_copy['hcgroup4'], sub_copy['comp_var_34'])
print (ct1)
# column percentages
colsum=ct1.sum(axis=0)
colpct=ct1/colsum
print(colpct)
# chi-square
print ('chi-square value for 3-4 pair, p value, expected counts')
cs1= scipy.stats.chi2_contingency(ct1)
print (cs1)

# END