# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 14:51:56 2015

@author: jesusfbes
"""

import pandas as pd
import numpy as np
import csv

from sklearn.cross_validation import train_test_split
from simple_regression import simple_regression
from sklearn.preprocessing import Imputer

def save_data(X_tr, y_tr, X_tst, y_tst):
    with open('data_train.csv', 'wb') as f:
        wtr = csv.writer(f, delimiter= ',')        
        wtr.writerows(np.hstack((X_tr,y_tr))) 
        
        
    with open('data_test.csv','wb') as f:
         wtr = csv.writer(f, delimiter= ',')
         wtr.writerows(X_tst) 
    
    with open('solutions.csv','wb') as f:
        wtr = csv.writer(f, delimiter= ',')
        wtr.writerow(['id', 'prediction'])
        for i,y in enumerate(y_tst):
            wtr.writerow([i,y[0]])
    
    with open('sample.csv','wb') as f:
        wtr = csv.writer(f, delimiter= ',')
        wtr.writerow(['id', 'prediction'])
        for i,y in enumerate(y_tst):
            wtr.writerow([i,0.0])
            
np.random.seed(10)


data = pd.read_excel('Folds5x2_pp.xlsx')

X = data.values[:,0:-1]
y = data.values[:,-1,np.newaxis]
# train-test split 
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.5,
                                                    random_state=10)
                                                    
# Drop all train data but 2000 samples 
N_train = 2000                                                    
X_train = X_train[0:N_train,:]
y_train = y_train[0:N_train,:]

(N_test,D) = np.shape(X_test)



# CASE 1_ no transform, no noise, no missing
print "CASE 1: no transform, no noise, no missing"

simple_regression(X_train, y_train, X_test, y_test)


# CASE 2 Noise added to X 
print " "
print "CASE 2: Noise added to X"

noise_train = np.random.normal(0,1,(N_train,D))
X_train = X_train + noise_train
noise_test = np.random.normal(0,1,(N_test,D))
X_test = X_test + noise_test

simple_regression(X_train, y_train, X_test, y_test)

# CASE 3 Nose added to X + one variable removed
#print "%%%%%%%%%%%%%%%%%%%%%"
#print "CASE 3: Noise added to X and one variable removed"
#print "1st"
#simple_regression(X_train[:,1:], y_train, X_test[:,1:], y_test)
#print "2nd"
#simple_regression(X_train[:,[0,2,3]], y_train, X_test[:,[0,2,3]], y_test)
#print "3rd"
#simple_regression(X_train[:,[0,1,3]], y_train, X_test[:,[0,1,3]], y_test)
#print "4th"
#simple_regression(X_train[:,[0,1,2]], y_train, X_test[:,[0,1,2]], y_test)


# CASE 4: Noise added to  X + 1st variable with missing data
print " "
print "CASE 4: Noise added to X and variable 1 with nan"
elements_to_nan_train = np.random.permutation(N_train)[:N_train/5]
X_train[elements_to_nan_train,0] = np.nan
elements_to_nan_test = np.random.permutation(N_test)[:N_test/5]
X_test[elements_to_nan_test,0] = np.nan

print "-> 1st variable removed"
simple_regression(X_train[:,1:], y_train, X_test[:,1:], y_test)

print "-> Impute the mean"
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X_train_imp = imp.fit_transform(X_train)
X_test_imp = imp.transform(X_test)
simple_regression(X_train_imp, y_train, X_test_imp, y_test)

# CASE 5: Noise added to  X + 1st variable with missing data 
# + additional noisy variable
print " "
print "CASE 5: Noise added to X and variable 1 with nan and noisy var"
noisy_variable_train = np.random.uniform(0,500,(N_train,1))

noisy_variable_test = np.random.uniform(0,500,(N_test,1))
X_train = np.hstack((X_train,noisy_variable_train))
X_test = np.hstack((X_test,noisy_variable_test))

save_data(X_train, y_train, X_test, y_test)

print "-> 1st variable removed"
simple_regression(X_train[:,1:], y_train, X_test[:,1:], y_test)

print "-> Impute the mean"
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X_train_imp = imp.fit_transform(X_train)
X_test_imp = imp.transform(X_test)
simple_regression(X_train_imp, y_train, X_test_imp, y_test)



    