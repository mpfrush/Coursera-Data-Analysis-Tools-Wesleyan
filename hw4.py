# -*- coding: utf-8 -*-
"""
Created on Sun Nov 8 2015

@author: violetgirl
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
import scipy
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


# ANOVA
print("\n ANOVA TEST")

# EMPLOYMENT RATE
# group the data in 2 groups and record it into new variable ecgroup2
def ecgroup2 (row):
    if row['employrate'] < 51:
        return 1 # unemployed or partially employed 
    elif row['employrate'] >= 51:
        return 2 # employed 
sub_copy['ecgroup2'] = sub_copy.apply(lambda row:  ecgroup2 (row), axis=1)

# HIV RATE
# group the data in 2 groups and record it into new variable hcgroup4
def hcgroup4 (row):
    if row['hivrate'] < 1.3:
        return 1 # not infected with HIV
    elif row['hivrate'] >= 1.3:
        return 2 # infected with HIV
sub_copy['hcgroup4'] = sub_copy.apply(lambda row:  hcgroup4 (row), axis=1)

## create datasets with the response and explanatory variables
## association between breast cancer rate and HIV rate
sub_b = sub_copy[['breastcancerper100th','hcgroup4']].dropna()

# using ols function for calculating the F-statistic and associated p value
print("\nBreast cancer rate vs. HIV rate")
model_b = smf.ols(formula='breastcancerper100th ~ C(hcgroup4)',data=sub_b).fit()
print(model_b.summary())
print("\nMean for the breast cancer rate by HIV infection status")
mb = sub_b.groupby('hcgroup4').mean()
print(mb)

sub_copy_cut = sub_copy[['breastcancerper100th','hcgroup4','ecgroup2']].dropna()
sub_u=sub_copy_cut[(sub_copy_cut['ecgroup2']==1)] # unemployed or partially employed
sub_e=sub_copy_cut[(sub_copy_cut['ecgroup2']==2)] # employed

print("\nBreast cancer rate vs. HIV rate for unemployed people")
model_u = smf.ols(formula='breastcancerper100th ~ C(hcgroup4)',data=sub_u).fit()
print(model_u.summary())

print("\nBreast cancer rate vs. HIV rate for employed people")
model_e = smf.ols(formula='breastcancerper100th ~ C(hcgroup4)',data=sub_e).fit()
print(model_e.summary())

print("\nMean for the breast cancer rate by HIV infection status for unemployed people")
mu = sub_u.groupby('hcgroup4').mean()
print(mu)

print("\nMean for the breast cancer rate by HIV infection status for employed people")
me = sub_e.groupby('hcgroup4').mean()
print(me)

# CHISQ TEST
print("\n CHISQ TEST")

# Breast Cancer Rate
# group the data in 2 groups and record it into new variable bcgroup4
def bcgroup4 (row):
    if row['breastcancerper100th'] < 50.3:
        return 1 
    elif row['breastcancerper100th'] >= 50.3:
        return 2 
sub_copy['bcgroup4'] = sub_copy.apply(lambda row:  bcgroup4 (row), axis=1)

# contingency table of observed counts
ct1=pd.crosstab(sub_copy['hcgroup4'], sub_copy['ecgroup2'])
print(ct1)
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
sub_copy['ecgroup2']=sub_copy['ecgroup2'].astype('category')
# set response variable to numeric
sub_copy['hcgroup4']=sub_copy['hcgroup4'].convert_objects(convert_numeric=True)

sub_copy_cut2 = sub_copy[['bcgroup4','hcgroup4','ecgroup2']].dropna()
sub_lb=sub_copy_cut2[(sub_copy_cut2['bcgroup4']==1)] # low breast cancer rate
sub_hb=sub_copy_cut2[(sub_copy_cut2['bcgroup4']==2)] # high breast cancer rate

# contingency table of observed counts
ct2=pd.crosstab(sub_lb['hcgroup4'], sub_lb['ecgroup2'])
print(ct2)
# column percentages
colsum=ct2.sum(axis=0)
colpct=ct2/colsum
print(colpct)
# chi-square
print ('chi-square value, p value, expected counts for those with low cancer rate')
cs2= scipy.stats.chi2_contingency(ct2)
print(cs2)

# contingency table of observed counts
ct3=pd.crosstab(sub_hb['hcgroup4'], sub_hb['ecgroup2'])
print(ct3)
# column percentages
colsum=ct3.sum(axis=0)
colpct=ct3/colsum
print(colpct)
# chi-square
print ('chi-square value, p value, expected counts for those with high cancer rate')
cs3= scipy.stats.chi2_contingency(ct3)
print(cs3)

plt.figure(1)
sb.factorplot(x='ecgroup2',y='hcgroup4',data=sub_lb,kind="point",ci=None)
plt.xlabel("Employment rate")
plt.ylabel("HIV rate")
plt.title('association between employment rate and HIV rate for those with low cancer rate')

plt.figure(2)
sb.factorplot(x='ecgroup2',y='hcgroup4',data=sub_hb,kind="point",ci=None)
plt.xlabel("Employment rate")
plt.ylabel("HIV rate")
plt.title('association between employment rate and HIV rate for those with high cancer rate')

# CORRELATION
print("\n CORRELATION TEST")

sub_copy_cut3 = sub_copy[['breastcancerper100th','hivrate','ecgroup2']].dropna()
sub_u3=sub_copy_cut3[(sub_copy_cut3['ecgroup2']==1)] # unemployed or partially employed
sub_e3=sub_copy_cut3[(sub_copy_cut3['ecgroup2']==2)] # employed

print ('association between breast cancer rate and HIV rate for unemployed people')
print('r coefficient and p value')
print (scipy.stats.pearsonr(sub_u3['breastcancerper100th'],sub_u3['hivrate']))

print ('association between breast cancer rate and HIV rate for employed people')
print('r coefficient and p value')
print (scipy.stats.pearsonr(sub_e3['breastcancerper100th'],sub_e3['hivrate']))

plt.figure(4)
sb.regplot(x="hivrate",y="breastcancerper100th",fit_reg=False,data=sub_u3)
plt.xlabel('HIV Rate')
plt.ylabel('Breast Cancer Rate')
plt.title('Breast Cancer Rate vs. HIV Rate for Unemployed People')

plt.figure(5)
sb.regplot(x="hivrate",y="breastcancerper100th",fit_reg=False,data=sub_e3)
plt.xlabel('HIV Rate')
plt.ylabel('Breast Cancer Rate')
plt.title('Breast Cancer Rate vs. HIV Rate for Employed People')

# END