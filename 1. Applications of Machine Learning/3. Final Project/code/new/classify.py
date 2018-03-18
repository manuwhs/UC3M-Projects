import pandas as pd
import numpy as np
import utilities_2 as util2
import utilities_5 as util5
import pylab as pl

import classifiers as cl

from sklearn.metrics import roc_auc_score

load_flag = 0;
if (load_flag == 1):
    X = util2.load_pickle ("X")    # Stored as a list
    Bidder_timeType = util2.load_pickle ("Bidder_timeType")
    Bidders_dic = util2.load_pickle ("Bidders_dic")  # We stored the dictionary in a list
    train_selection = util2.load_pickle ("train_selection")  # Selected training bidders
    test_selection = util2.load_pickle ("test_selection")  # Selected training bidders
    
    
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    
    train_selection = train_selection[0]
    test_selection = test_selection[0]
    test_selection.index = range(len(test_selection)) # Fix index for following operations
    Bidders_dic = Bidders_dic[0]
    X = np.array(X)   # Vector of features for all bidders


#==============================================================================
#     # Obtain the ones just for the Selected train 
#==============================================================================
    n_sa, n_features = X.shape
    xtrain = np.zeros((len(train_selection),n_features))
    ytrain = np.zeros(len(train_selection))
     
    for i in range (len(train_selection)):
        indx = Bidders_dic[train_selection.bidder_id[i]]
        ytrain[i] = train_selection.outcome[i]
        xtrain[i] = X[indx][:]
#==============================================================================
#     # Obtain the ones just for the Selected test 
#==============================================================================
    n_sa, n_features = X.shape
    xtest = np.zeros((len(test_selection),n_features))

    for i in range (len(test_selection)):
        indx = Bidders_dic[test_selection.bidder_id[i]]
        xtest[i] = X[indx][:]
        
        
#==============================================================================
#==============================================================================
#==============================================================================
#                           Preprocess
#==============================================================================
#==============================================================================
#==============================================================================

PP_flag = 0;
if (PP_flag == 1):
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(X)
    xtrain = scaler.transform(xtrain)            
    xtest = scaler.transform(xtest)                               

#==============================================================================
#       CREATE CLASSIFIERS !!
#==============================================================================

#scores = cl.svm(xtrain,ytrain,xtest,kernel='linear',model_eval='recall',ensemble=None,soft_output=True,verbose=True);
#scores_tr = cl.svm(xtrain,ytrain,xtrain,kernel='linear',model_eval='recall',ensemble=None,soft_output=True,verbose=True);

#classifier = cl.get_svm(xtrain,ytrain)
#classifier.fit(xtrain, ytrain)
#score_final_tr = classifier.predict_proba(xtrain)
#score_final_tr = score_final_tr[:,1]

scores = classifier.predict_proba(xtest)
scores = scores[:,1]

#ntrain,nfeatures = xtrain.shape
#ntest,nfeatures = xtest.shape
#
#bag_traininig, ylabels = cl.get_bagger_xtrain(xtrain,ytrain, 0.9, 2)
#classifier = cl.get_svm(xtrain,ytrain)
#
#n_baggers = 50;
#
#scores = np.zeros((ntest,n_baggers));
#scores_tr = np.zeros((ntrain,n_baggers));
#
#for i in range (n_baggers):
#    print i 
#    # Obtain bag and train classifier
#    bag_traininig, ylabels = cl.get_bagger_xtrain(xtrain,ytrain, 0.8, 4)
##    classifier = cl.get_svm(bag_traininig, ylabels)
#    classifier.fit(bag_traininig, ylabels)
#    
#    output = classifier.predict_proba(xtrain)
#    scores_tr[:,i] = output[:,1]
#    
#    output = classifier.predict_proba(xtest)
#    scores[:,i] = output[:,1]


#weights = np.dot(np.linalg.pinv(scores_tr),ytrain)
#score_final_tr =  np.dot(scores_tr,weights)
#
#score_final_tr =  np.mean(scores_tr, axis = 1)
#score_final =  np.mean(scores, axis = 1)


#scores_tr /= n_baggers;
#scores /= n_baggers;

n_error = 0.0;

for i in range (len(score_final_tr)):
    if (score_final_tr[i] > 0.5 ):
        if (ytrain[i] < 0.5 ):
            n_error += 1
            
    if (score_final_tr[i] < 0.5 ):
        if (ytrain[i] > 0.5 ):
            n_error += 1

acc = n_error/len(score_final_tr)
print acc

auc = roc_auc_score(ytrain,score_final_tr)
print auc
#==============================================================================
# WRITE OUTPUT FILE
#==============================================================================


write_out_flag = 1;
if (write_out_flag == 1):
    results = np.zeros(len(test))
    
    for i in range(len(test_selection)):
        pos = test.loc[test['bidder_id'] == test_selection.bidder_id[i]].index[0]
        results[pos] = scores[i]
    
#        if (results[pos] < 0.5):
#            results[pos] += results[pos] * 0.5
    
        
    util5.write_csv_output(results, "pene.csv")



#for i in range (n_baggers):
#    print i 
#    # Obtain bag and train classifier
#    bag_traininig, ylabels = cl.get_bagger_xtrain(xtrain,ytrain, 0.8, 4)
#    classifier.fit(xtrain,ytrain)
#    
#    output = classifier.predict_proba(xtrain)
#    scores_tr[:,i] = output[:,1]
#    
#    output = classifier.predict_proba(xtest)
#    scores[:,i] = output[:,1]
#
#weights = np.dot(np.linalg.pinv(scores_tr),ytrain)
#
#score_final =  np.dot(scores_tr,weights)
##scores_tr /= n_baggers;
##scores /= n_baggers;
#
#n_error = 0.0;
