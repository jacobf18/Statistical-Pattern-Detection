#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

dataframe = pd.read_csv('VIX.csv')

adj_close = dataframe['Adj Close']

binary = np.empty([0])
for i, val in enumerate(adj_close):
    up = 0
    if i == len(adj_close)-1:
        break
    if adj_close[i+1] - val < 0:
        up = 0
    else:
        up = 1
    binary = np.append(binary, [up])

ones = np.sum(a=binary)
zeroes = len(binary) - ones

z_score = 1.96

mean = (2 * ones * zeroes / (ones + zeroes)) + 1
variance = (mean - 1) * (mean - 2) / (ones+zeroes-1)

num_runs = 0
prev_val = 0
for val in binary:
    if val != prev_val:
        num_runs += 1
        prev_val = val

predicted_runs = z_score * (variance**0.5) + mean

predicted_z = abs(num_runs - mean) / (variance**0.5)

print('Actual Runs: ' + str(num_runs))
print('Predicted Runs: ' + str(predicted_runs))
print('Predicted Z-Score: ' + str(predicted_z))
print('Mean: ' + str(mean))
print('Variance: ' + str(variance))
print('Z-Score: ' + str(z_score))

