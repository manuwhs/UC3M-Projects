
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.style.use('ggplot')


from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale
from sklearn.feature_selection import GenericUnivariateSelect,f_classif
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score


# In[2]:

# Load test users
test_users = pd.read_csv('data/test.csv',index_col=0)


# In[3]:

# Load training data
xtrain = pd.read_csv('data/train_data_nogap.csv',index_col=0)
ytrain = pd.Series.from_csv('data/labels.csv',index_col=0)
xtest = pd.read_csv('data/test_data_nogap.csv',index_col=0)


# In[4]:

# Delete users with NaN features
nan_idx = pd.isnull(xtrain).any(axis=1)
xtrain = xtrain.ix[~nan_idx.values,:]
ytrain = ytrain[~nan_idx.values].values
nan_idx = pd.isnull(xtest).any(axis=1)
xtest = xtest.ix[~nan_idx.values,:]


# In[5]:

#==============================================================================
#==============================================================================
#==============================================================================
#                           Preprocess
#==============================================================================
#==============================================================================
#==============================================================================
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(xtrain)
xtrain = scaler.transform(xtrain)            
xtest = scaler.transform(xtest)
xtrain = pd.DataFrame(xtrain)
xtest = pd.DataFrame(xtest)

# Create one-vs-all output matrix
ytrain_m = np.zeros((len(ytrain),2))
ytrain_m[np.where(ytrain==0),0] = True
ytrain_m[np.where(ytrain==1),1] = True


# In[6]:

#==============================================================================
#                           Feature Extraction
#==============================================================================

train_idx = xtrain.index.values
train_hdr = xtrain.columns.values
test_idx = xtest.index.values
test_hdr = xtest.columns.values

FE_flag = 1;
if (FE_flag == 1):
    from sklearn.decomposition import PCA,KernelPCA
    from sklearn.cross_decomposition import PLSSVD,PLSCanonical,PLSRegression
    from sklearn.cross_decomposition import CCA
    from sklearn.lda import LDA
    import sklearn.metrics.pairwise as pair
    from sklearn.preprocessing import KernelCenterer
    from sklearn.kernel_approximation import Nystroem,RBFSampler
    
    n_comp = 15
    n_class = 2
    #==============================================================================
    # Linear PCA
    FE_PCA = 0
    if (FE_PCA == 1):
        pca = PCA(n_components=n_comp)
        xtrain = pca.fit_transform(xtrain)
        xtest = pca.transform(xtest)
    
    
    # Linear PLS2
    FE_PLS = 0
    if (FE_PLS == 1):
        pls2 = PLSRegression(n_components=n_comp)
        pls2.fit(xtrain,ytrain_m)
        xtrain = pls2.transform(xtrain)
        xtest = pls2.transform(xtest)
        
        
    # Linear CCA
    FE_CCA = 0
    if (FE_CCA == 1):
        cca = CCA(n_components=n_class)
        cca.fit(xtrain,ytrain_m)
        xtrain = cca.transform(xtrain)
        xtest = cca.transform(xtest)
        
        
    # Kernel PCA
    FE_kPCA = 0
    if (FE_kPCA == 1):
        d = pair.pairwise_distances(xtrain,xtrain)
        aux = np.triu(d)
        sigma = np.sqrt(np.mean(np.power(aux[aux!=0],2)*0.5))
        gamma = 1/(2*sigma**2)
        kpca = KernelPCA(n_components=n_comp,kernel="rbf",gamma=gamma)
        xtrain = kpca.fit_transform(xtrain)
        xtest = kpca.transform(xtest)
        
    # Kernel PLS
    FE_kPLS = 1
    if (FE_kPLS == 1):
        d = pair.pairwise_distances(xtrain,xtrain)
        aux = np.triu(d)
        sigma = np.sqrt(np.mean(np.power(aux[aux!=0],2)*0.5))
        gamma = 1/(2*sigma**2)
        ktrain = pair.rbf_kernel(xtrain,xtrain,gamma)
        ktest = pair.rbf_kernel(xtest,xtrain,gamma)
        kcent = KernelCenterer()
        kcent.fit(ktrain)
        ktrain = kcent.transform(ktrain)
        ktest = kcent.transform(ktest)
        kpls = PLSRegression(n_components=n_comp)
        kpls.fit(ktrain,ytrain_m)
        xtrain = kpls.transform(ktrain)
        xtest = kpls.transform(ktest)
        
    # LDA
    FE_LDA = 0
    if (FE_LDA == 1):
        lda = LDA()
        lda.fit(xtrain,ytrain)
        xtrain = lda.transform(xtrain)
        xtest = lda.transform(xtest)

    xtrain = pd.DataFrame(xtrain,index=train_idx)
    xtest = pd.DataFrame(xtest,index=test_idx)


# In[7]:

def balanced_sets(xtrain,ytrain):
    nbots = sum(ytrain==1)
    nhumans = sum(ytrain==0)
    nfolds = nhumans/nbots
    yhumans_idx = np.where(ytrain==0)[0]
    np.random.shuffle(yhumans_idx)
    res = []
    for i in np.arange(nfolds):
        selector = yhumans_idx[i*nbots:(i+1)*nbots]
        xtrain_res = np.r_[xtrain.ix[selector,:],xtrain.ix[ytrain==1,:]]
        ytrain_res = np.r_[ytrain[selector],ytrain[ytrain==1]]
        res.append((xtrain_res,ytrain_res))
        
    return res

train_groups = balanced_sets(xtrain,ytrain)


# In[9]:

# Random Forest Classifier with balanced Baggers (Training)
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
yval_prob = []
yval_set = []
for s in train_groups:
    xt = s[0]
    yt = s[1]
    xttrain,xval,yttrain,yval = train_test_split(xt,yt,test_size=0.2,random_state=0)
    et = ExtraTreesClassifier(n_estimators=1000,criterion='gini',class_weight='auto',n_jobs=-1)
    et.fit(xttrain,yttrain)
    yval_set.append(yval)
    yval_prob.append(et.predict_proba(xval))
    
yval_prob = np.mean(yval_prob,axis=0)
for s in yval_set:
    auc = roc_auc_score(s,yval_prob[:,et.classes_==1])
    print 'Random Forest with BBagging: '+str(auc)


# In[10]:

# Random Forest Classifier with balanced Baggers (Test) 
ytest_prob = []
for s in train_groups:
    xt = s[0]
    yt = s[1]
    et = ExtraTreesClassifier(n_estimators=1000,criterion='gini',class_weight='auto',n_jobs=-1)
    et.fit(xt,yt)
    ytest_prob.append(et.predict_proba(xtest))
    
scores = np.mean(ytest_prob,axis=0)


# In[12]:

# Write results into file
result = pd.Series(np.zeros(len(test_users)),index=test_users.index)
result.loc[xtest.index] = scores[:,1]
test_users['prediction'] = result
test_users.to_csv('result.csv',columns=['prediction'],index_label='bidder_id')

