# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:28:40 2015

@author: Jesus Fernandez Bes, jesusfbes@tsc.uc3m.es
"""


import pandas as pd
import numpy as np
import csv

import matplotlib.pyplot as plt
from sklearn import preprocessing

from scipy import spatial

# Load data
f_train = 'data_train.csv'
f_test = 'data_test.csv'

output_file = 'GP.csv'

data_train = pd.read_csv(f_train, header = None)
X = data_train.values[:,:-1]
Ytrain = data_train.values[:,-1]
Nsamples, Ndim = X.shape
Xtest = pd.read_csv(f_test, header = None).values

# Fill the nan with the average 
for d in range(Ndim):
    # For the training
    not_nan_index = np.where(np.isnan(X[:,d]) == False)
    not_nan = X[not_nan_index,d]
    
    average = np.mean(not_nan)
    
    nan_index = np.where(np.isnan(X[:,d]) == True)
    for i in nan_index:
        X[i,d] = average

for d in range(Ndim):
    # For the testing
    not_nan_index = np.where(np.isnan(Xtest[:,d]) == False)
    not_nan = Xtest[not_nan_index,d]
    
    average = np.mean(X[:,d])
    
    nan_index = np.where(np.isnan(Xtest[:,d]) == True)
    for i in nan_index:
        Xtest[i,d] = average


Xtrain = X

#%% Normalize data
scaler = preprocessing.StandardScaler().fit(Xtrain)
Xtrain = scaler.transform(Xtrain)            
Xtest = scaler.transform(Xtest)   


# Configuro los valores de sigma_eps y l (cogidos por defecto de los apuntes)
sigma_eps = 0.3
l = 0.2

# Calculo las distancias entre los valores de X_train y X_test    
dist = spatial.distance.cdist(Xtrain,Xtrain,'euclidean')
dist_ss = spatial.distance.cdist(Xtest,Xtest,'euclidean')
dist_s = spatial.distance.cdist(Xtest,Xtrain,'euclidean')

# Calculo las covarianzas con las distancias obtenidas anteriormente
K = np.exp(-np.power(dist,2)/(2*l))
K_ss = np.exp(-np.power(dist_ss,2)/(2*l))
K_s = np.exp(-np.power(dist_s,2)/(2*l))
 
# Calculo la media y la covarianza de f*   
m = K_s.dot(np.linalg.inv(K + sigma_eps**2 * np.eye(2000))).dot(Ytrain)
Cov = K_ss - K_s.dot(np.linalg.inv(K + sigma_eps**2 * np.eye(2000))).dot(K_s.T)

# Calculo f* con la media y la covarianza obtenidas anteriormente
f_estrella=np.random.multivariate_normal(m,Cov)


Otest = m


# Save output file
with open(output_file,'wb') as f:
        wtr = csv.writer(f, delimiter= ',')
        wtr.writerow(['id', 'prediction'])
        for i,x in enumerate(Otest):
            wtr.writerow([i,x])
            
print "Saved Baseline solution"

