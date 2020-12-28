#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 15:29:10 2020

@author: ivanpatel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

# Data Source - https://www.openintro.org/data/index.php?data=pew_energy_2018
data = pd.read_csv('/Users/ivanpatel/Desktop/pew_energy_2018.csv')

# EDA
data.columns # 6 columns about different energy sources
print(data.head(), '\n'); data.info() # All columns are categorical. No Null Values

# Thus, we can calculate the proportions of each answer for a given energy source

# A function that shows a bar plot of responses for a given energy source
def plot_responses(column_name, energy_source):
    
    # Get the total Observations
    total_obvs = np.sum(data[column_name].value_counts().values)
    
    # Make the figure
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(list(data[column_name].value_counts().keys()), list(data[column_name].value_counts().values / total_obvs))
    ax.set(xlabel = 'Support', ylabel = '% of Responses', title = 'Support for ' + energy_source)
    plt.show()

# Plot all the energy sources
for i in data.columns:
    plot_responses(i, i)

'''
I am interested in looking at offshore_drilling, coal_mining, and solar panels.
 
Given the data, we can use hypothesis testing to answer three questions:
    1. Do majority of Americans favor the increase in offshore drilling?
    
    2. Do majority of Americans oppose or favor the expansion of coal mining 
       to produce energy?
       
    3. Do majority of American favor the increase the usage of solar panel farms
       to produce energy. 
       
I'll test the first hypothesis:

    If p is the proportion of americans favor offshore_drilling, then:
        
    Null Hypothesis: p = 0.5
    Alternative Hypothesis: p < 0.5
    
B/c the our sample size is large, and the samples were randomly seleted, 
the distribution of p-hat under the CLT is normal. 
'''
p_hat = data['offshore_drilling'].value_counts().values[1] / np.sum(data['offshore_drilling'].value_counts().values)
std_error = np.sqrt(0.5**2/ data.shape[0])         

# p-value: http://www.z-table.com/
z_score = (p_hat - 0.5) / std_error 

print(p_hat)
print(std_error)
print(z_score)