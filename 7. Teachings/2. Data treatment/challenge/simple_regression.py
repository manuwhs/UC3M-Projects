# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 15:45:31 2015

@author: jesusfbes
"""

import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def simple_regression(X_train, y_train, X_test, y_test):
    
    # BASELINE REGRESSOR
    y_baseline1 =  np.mean(y_train)

    error_baseline1_tst = np.sqrt( np.mean((y_test - y_baseline1)**2) )
    print 'error baseline: ' + str(error_baseline1_tst)

  
  # STANDARD LINEAR REGRESSSOR
    reg =  LinearRegression()
    cv_mse = -1 * cross_val_score(reg, X_train, y_train, cv=5,
                                  scoring='mean_squared_error')         
   
    print 'error cv linear: ' + str(np.sqrt( np.mean(cv_mse)))
    reg =  LinearRegression()
    reg.fit(X_train,y_train)
    y_linear1 = reg.predict(X_test) 
    error_linear1_tst = np.sqrt( np.mean((y_test - y_linear1)**2) )
    print 'error test linear: ' + str(error_linear1_tst)
  

    # STANDARD LINEAR REGRESSSOR (Normalized) 
    scaler = StandardScaler()
    X_train_n = scaler.fit_transform(X_train)
    X_test_n = scaler.transform(X_test)
    reg =  LinearRegression()
    cv_mse = -1 * cross_val_score(reg, X_train_n, y_train, cv=5,
                                  scoring='mean_squared_error')     
    print 'error cv linear (standarized): ' + str(np.sqrt( np.mean(cv_mse)))                              
    reg =  LinearRegression()
    reg.fit(X_train_n, y_train)
    y_linear2 = reg.predict(X_test_n) 
    error_linear2 = np.sqrt( np.mean((y_test - y_linear2)**2) )
    print 'error test linear (standarized): ' + str(error_linear2)
    
    
    #RANDOM FOREST
    reg = RandomForestRegressor(n_estimators=100,max_features='log2')
    cv_mse = -1 * cross_val_score(reg, X_train_n, y_train.ravel(), cv=5,
                                  scoring='mean_squared_error')      
    print 'error cv RF (standarized): ' + str(np.sqrt( np.mean(cv_mse)))  
    reg = RandomForestRegressor(n_estimators=100,max_features='log2')
    reg.fit(X_train_n,y_train.ravel())
    y_rf = reg.predict(X_test_n)
    error_rf = np.sqrt( np.mean((y_test.ravel() - y_rf)**2) )
    print 'error test RF: ' + str(error_rf)