import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier

from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

def svm(xTrain,yTrain,xTest,kernel='linear',n_baggers = 100, model_eval='recall',ensemble=None,soft_output=True,verbose=True):
    
    # DEFINIMOS PARAMETROS SVM: CROSVALIDATION DE PARAMETROS DE LA SVM
    # Check params
    if ensemble!=None:
        if ensemble!='bagging':
            print('Ensemble method not recognized. Supported value is: "bagging"')
            return
    
    nSamples,nFeatures = np.shape(xTrain)
    C = np.logspace(-3,3,10)
    param_grid = dict()
    if kernel=='linear':
        if verbose:
            print('Training linear SVM classifier')
        param_grid.update({'kernel':['linear']})
    elif kernel=='poly':
        if verbose:
            print('Training polynomic SVM classifier')
        degree = np.arange(2,5)
        param_grid.update({'kernel':['poly']})
        param_grid.update({'degree':degree})
    elif kernel=='rbf':
        if verbose:
            print('Training gaussian SVM classifier')
        gamma = np.array([0.125,0.25,0.5,1,2,4])/nFeatures
        param_grid.update({'kernel':['rbf']})
        param_grid.update({'gamma':gamma})
    else:
        print('Kernel not recognized. Supported values are: "linear", "poly" or "rbf"')
        return
    param_grid.update({'C':C})
    if soft_output:
        param_grid.update({'probability':[True]})
    
    stkfold = StratifiedKFold(yTrain,n_folds=5)
    gs = GridSearchCV(SVC(class_weight='auto'),param_grid,scoring=model_eval,cv=stkfold,refit=True,n_jobs=-1)
    gs.fit(xTrain,yTrain)
    est = gs.best_estimator_
    if verbose:
            print('Best classifier:')
            print(est)
    
    
    # UTILIZAR LA SVM YA VALIDADA CON BAGGING O NO
    
    
    if ensemble=='bagging':
        if verbose:
            print('Training bagging estimator')
        bag = BaggingClassifier(est,n_estimators = n_baggers,oob_score=True,n_jobs=-1)
        bag.fit(xTrain,yTrain)
        if soft_output:
            scores = bag.predict_proba(xTest)
        else:
            scores = bag.predict(xTest)
#        print('Accuracy: '+str(bag.score(xTest,yTest)))   # No disponemos de las putas yTest
    else:
        if soft_output:
            scores = est.predict_proba(xTest)
        else:
            scores = est.predict(xTest)
#        print('Accuracy: '+str(est.score(xTest,yTest)))  # No disponemos de las putas yTest
    
    return scores

def get_svm(xTrain,yTrain,kernel='linear',n_baggers = 100, model_eval='recall',ensemble=None,soft_output=True,verbose=True):
    
    # DEFINIMOS PARAMETROS SVM: CROSVALIDATION DE PARAMETROS DE LA SVM
    # Check params
    if ensemble!=None:
        if ensemble!='bagging':
            print('Ensemble method not recognized. Supported value is: "bagging"')
            return
    
    nSamples,nFeatures = np.shape(xTrain)
    C = np.logspace(-3,3,10)
    param_grid = dict()
    if kernel=='linear':
        if verbose:
            print('Training linear SVM classifier')
        param_grid.update({'kernel':['linear']})
    elif kernel=='poly':
        if verbose:
            print('Training polynomic SVM classifier')
        degree = np.arange(2,5)
        param_grid.update({'kernel':['poly']})
        param_grid.update({'degree':degree})
    elif kernel=='rbf':
        if verbose:
            print('Training gaussian SVM classifier')
        gamma = np.array([0.125,0.25,0.5,1,2,4])/nFeatures
        param_grid.update({'kernel':['rbf']})
        param_grid.update({'gamma':gamma})
    else:
        print('Kernel not recognized. Supported values are: "linear", "poly" or "rbf"')
        return
    param_grid.update({'C':C})
    if soft_output:
        param_grid.update({'probability':[True]})
    
    stkfold = StratifiedKFold(yTrain,n_folds=5)
    gs = GridSearchCV(SVC(class_weight='auto'),param_grid,scoring=model_eval,cv=stkfold,refit=True,n_jobs=-1)
    gs.fit(xTrain,yTrain)
    est = gs.best_estimator_

    return est
    

def logistic_regression(xTrain,yTrain,xTest,model_eval='recall',ensemble=None,soft_output=True,verbose=True):
    
    # Check params
    if ensemble!=None:
        if ensemble!='bagging':
            print('Ensemble method not recognized. Supported value is: "bagging"')
            return
    
    nSamples,nFeatures = np.shape(xTrain)
    C = np.logspace(-3,3,10)
    param_grid = dict()
    param_grid.update({'C':C})
    
    if verbose:
        print('Training logistic regression classifier')
    stkfold = StratifiedKFold(yTrain,n_folds=5)
    gs = GridSearchCV(LogisticRegression(class_weight='auto'),param_grid,scoring=model_eval,cv=stkfold,refit=True,n_jobs=-1)
    gs.fit(xTrain,yTrain)
    est = gs.best_estimator_
    
    if ensemble=='bagging':
        if verbose:
            print('Training bagging estimator')
        bag = BaggingClassifier(est,n_estimators=1000,oob_score=True,n_jobs=-1)
        bag.fit(xTrain,yTrain)
        if soft_output:
            scores = bag.predict_proba(xTest)
        else:
            scores = bag.predict(xTest)
#        print('Accuracy: '+str(bag.score(xTest,yTest)))
    else:
        if soft_output:
            scores = est.predict_proba(xTest)
        else:
            scores = est.predict(xTest)
#        print('Accuracy: '+str(est.score(xTest,yTest)))
    
    return est
    

def random_forest(xTrain,yTrain,xTest,yTest,model_eval='recall',ensemble=None,soft_output=True,verbose=True):
    
    # Check params
    if ensemble!=None:
        if ensemble!='bagging':
            print('Ensemble method not recognized. Supported value is: "bagging"')
            return
    
    nSamples,nFeatures = np.shape(xTrain)
    C = np.logspace(-3,3,10)
    param_grid = dict()
    param_grid.update({'C':C})
    
    if verbose:
        print('Training logistic regression classifier')
    stkfold = StratifiedKFold(yTrain,n_folds=5)
    gs = GridSearchCV(LogisticRegression(class_weight='auto'),param_grid,scoring=model_eval,cv=stkfold,refit=True,n_jobs=-1)
    gs.fit(xTrain,yTrain)
    est = gs.best_estimator_
    
    if ensemble=='bagging':
        if verbose:
            print('Training bagging estimator')
        bag = BaggingClassifier(est,n_estimators=1000,oob_score=True,n_jobs=-1)
        bag.fit(xTrain,yTrain)
        if soft_output:
            scores = bag.predict_proba(xTest)
        else:
            scores = bag.predict(xTest)
        print('Accuracy: '+str(bag.score(xTest,yTest)))
    else:
        if soft_output:
            scores = est.predict_proba(xTest)
        else:
            scores = est.predict(xTest)
        print('Accuracy: '+str(est.score(xTest,yTest)))
    
    return scores
    
    
