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
from sklearn import cross_validation
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


#Xtrain = Xtrain[0:1000,:]   # Acortar para que sea mas rápido al hacer pruebas
#Ytrain = Ytrain[0:1000]

# Function that trains with Xtrain and gives the Output of Xval

def get_GP_O_val (Xtrain, Ytrain, Xval,sigma_eps, l ):       
    # Calculo las distancias entre los valores de X_train y X_test    
    dist = spatial.distance.cdist(Xtrain,Xtrain,'euclidean')
#    dist_ss = spatial.distance.cdist(Xtest,Xtest,'euclidean')
    dist_s = spatial.distance.cdist(Xval,Xtrain,'euclidean')
    
    # Calculo las covarianzas con las distancias obtenidas anteriormente
    K = np.exp(-np.power(dist,2)/(2*l))
#    K_ss = np.exp(-np.power(dist_ss,2)/(2*l))
    K_s = np.exp(-np.power(dist_s,2)/(2*l))
     
    # Calculo la media y la covarianza de f*   
    m = K_s.dot(np.linalg.inv(K + sigma_eps**2 * np.eye(Xtrain.shape[0]))).dot(Ytrain)
#            Cov = K_ss - K_s.dot(np.linalg.inv(K + sigma_eps**2 * np.eye(2000))).dot(K_s.T)
    
    # Calculo f* con la media y la covarianza obtenidas anteriormente
#            f_estrella=np.random.multivariate_normal(m,Cov)
    
    return m
    
# Barridillo de parámetros 
k_fold = 5
l_list = [0.3, 0.5, 0.6]
sigma_eps_list = [0.3, 4, 6 ,9]


# Configuro los valores de sigma_eps y l (cogidos por defecto de los apuntes)

Emse_all = np.zeros((len(l_list),len(sigma_eps_list)))

kfold = cross_validation.KFold(Ytrain.size, n_folds = k_fold)
for l_i in range (len(l_list)):   # For each parameter of the grid search
    for sigma_eps_i in range (len(sigma_eps_list)):
        
        print "Calculating for ", l_list[l_i], sigma_eps_list[sigma_eps_i]    
        Emse = 0
        for train_index, val_index in kfold:  # Perform 5 fold
            # Entrenar y obtener la salida para el set de validacion
#            print train_index.shape
            Oval = get_GP_O_val(Xtrain[train_index],Ytrain[train_index], Xtrain[val_index], l_list[l_i],sigma_eps_list[sigma_eps_i] )
            # Obtener el MSE del set de validacion
            Emse += np.mean(np.power(Oval - Ytrain[val_index],2))
        
        Emse_all[l_i][sigma_eps_i] = Emse / k_fold

## Obtain the combination of parameters for which Emse_all is minimum

j = np.argmin(Emse_all) # Gets the column of the minimum
i = np.argmin(Emse_all[:,j]) # Gets the row of the minimum

l = l_list[i]
sigma_eps = sigma_eps_list[j]

print l, sigma_eps
Otest = get_GP_O_val (Xtrain, Ytrain, Xtest, sigma_eps, l )



# Save output file
with open(output_file,'wb') as f:
        wtr = csv.writer(f, delimiter= ',')
        wtr.writerow(['id', 'prediction'])
        for i,x in enumerate(Otest):
            wtr.writerow([i,x])
            
print "Saved Baseline solution"

