# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:28:40 2015

@author: jesusfbes
"""


import pandas as pd
import numpy as np
import csv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor

f_train = 'data_train.csv'
f_test = 'data_test.csv'


data_train = pd.read_csv(f_train, header = None)
X_train = data_train.values[:,:-1]
y_train = data_train.values[:,-1]

X_test = pd.read_csv(f_test, header = None).values

# Generate Baseline
y_baseline =  np.mean(y_train)

# Save Baseline
with open('baseline.csv','wb') as f:
        wtr = csv.writer(f, delimiter= ',')
        wtr.writerow(['id', 'prediction'])
        for i,x in enumerate(X_test):
            wtr.writerow([i,y_baseline])
            
print "Saved Baseline solution"

# Generate linear without var1 that contains nans
reg =  LinearRegression()
reg.fit(X_train[:,1:],y_train)
y_linear1 = reg.predict(X_test[:,1:]) 

# Save Baseline
with open('linear_4vars.csv','wb') as f:
        wtr = csv.writer(f, delimiter= ',')
        wtr.writerow(['id', 'prediction'])
        for i,y in enumerate(y_linear1):
            wtr.writerow([i,y])
            
print "Saved linear solution"


# Generate RF with imputation in the values of var1
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X_train_imp = imp.fit_transform(X_train)
X_test_imp = imp.transform(X_test)
reg = RandomForestRegressor(n_estimators=100,max_features='log2')
reg.fit(X_train_imp,y_train.ravel())
y_rf = reg.predict(X_test_imp)

# Save Baseline
with open('rf_100.csv','wb') as f:
        wtr = csv.writer(f, delimiter= ',')
        wtr.writerow(['id', 'prediction'])
        for i,y in enumerate(y_rf):
            wtr.writerow([i,y])
            
print "Saved random forest solution"