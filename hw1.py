# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 2015

@author: violetgirl
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi

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
print("Statistics for a Suicide Rate")
print(data['suicideper100th'].describe())

# subset data for a high suicide rate based on summary statistics
sub = data[(data['suicideper100th']>12)]
#make a copy of my new subsetted data
sub_copy = sub.copy()
# remove missing values 
sub_copy=sub_copy.dropna()

# EMPLOYMENT RATE
# group the data in 4 groups and record it into new variable ecgroup4
def ecgroup4 (row):
    if row['employrate'] >= 32 and row['employrate'] < 51:
        return 1
    elif row['employrate'] >= 51 and row['employrate'] < 59:
        return 2
    elif row['employrate'] >= 59 and row['employrate'] < 65:
        return 3
    elif row['employrate'] >= 65 and row['employrate'] < 84:
        return 4

sub_copy['ecgroup4'] = sub_copy.apply(lambda row:  ecgroup4 (row), axis=1)

# create datasets with the response and explanatory variables
# breast cancer rate vs employment rate
sub_b = sub_copy[['breastcancerper100th','ecgroup4']].dropna()
# HIV rate vs employment rate
sub_h = sub_copy[['hivrate','ecgroup4']].dropna()

# usinf ols function for calculating the F-statistic and associated p value
model_b = smf.ols(formula='breastcancerper100th ~ C(ecgroup4)',data=sub_b)
results_b=model_b.fit()
mb = sub_b.groupby('ecgroup4').mean()
stb = sub_b.groupby('ecgroup4').std()
print("Breast cancer rate vs. employment rate")
print(results_b.summary())
print("Mean and standard deviation for the breast cancer rate by employment status")
print(mb,"\n\n",stb)

model_h = smf.ols(formula='hivrate ~ C(ecgroup4)',data=sub_h)
results_h=model_h.fit()
mh = sub_h.groupby('ecgroup4').mean()
sth = sub_h.groupby('ecgroup4').std()
print("HIV rate vs. employment rate")
print(results_h.summary())
print("Mean and standard deviation for the HIV rate by employment status")
print(mh,"\n\n",sth)

# post hoc test
print("Post Hoc Test for breast cancer rate")
mcb = multi.MultiComparison(sub_b['breastcancerper100th'],sub_b['ecgroup4'])
resb=mcb.tukeyhsd()
print(resb.summary())

print("Post Hoc Test for HIV rate")
mch = multi.MultiComparison(sub_h['hivrate'],sub_h['ecgroup4'])
resh=mch.tukeyhsd()
print(resh.summary())

# END