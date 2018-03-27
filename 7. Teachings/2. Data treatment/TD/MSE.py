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

# Load data
f_train = 'data_train.csv'
f_test = 'data_test.csv'

output_file = 'MSE.csv'

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

 #################################################################
 #################### DATA PREPROCESSING #########################
 #################################################################
 ## Split data in training and testing
#train_ratio = 0.9
#rang = np.arange(np.shape(X)[0],dtype=int) # Create array of index
#np.random.seed(0)
#rang = np.random.permutation(rang)        # Randomize the array of index
# 
#Ntrain = round(train_ratio*np.shape(X)[0])    # Number of samples used for training
#Ntest = len(rang)-Ntrain                  # Number of samples used for testing
# 
#Xtrain = X[rang[:Ntrain]]
#Xtest = X[rang[Ntrain:]]
# 
#Ytrain = Y[rang[:Ntrain]]
#Ytest = Y[rang[Ntrain:]]

Xtrain = X

#%% Normalize data
scaler = preprocessing.StandardScaler().fit(Xtrain)
Xtrain = scaler.transform(Xtrain)            
Xtest = scaler.transform(Xtest)   


#################################################################
#################### GMDH #########################
#################################################################

plt.close("all")

Ntrain, Ndim = Xtrain.shape
Ntest, Ndim = Xtest.shape

# Append ones
Xtrain = np.concatenate ((np.ones((Ntrain,1)),Xtrain),axis = 1)
Xtest = np.concatenate ((np.ones((Ntest,1)),Xtest),axis = 1)

########################################################################
print "PENE"
Xtrain = Xtrain.T
W = np.linalg.inv(Xtrain.dot(Xtrain.T))
W = W.dot(Xtrain).dot(Ytrain)

Otest = Xtest.dot(W)

# Save output file
with open(output_file,'wb') as f:
        wtr = csv.writer(f, delimiter= ',')
        wtr.writerow(['id', 'prediction'])
        for i,x in enumerate(Otest):
            wtr.writerow([i,x])
            
print "Saved Baseline solution"

