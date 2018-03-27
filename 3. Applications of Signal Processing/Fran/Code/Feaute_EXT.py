
# coding: utf-8

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#==============================================================================
#                     FLAGS
#==============================================================================
PREPROCESS_F = 0;  # Preprocessing (normalizing N(0,1)) the sample vectors

FE_PCA = 0
FE_PLS = 0
FE_CCA = 0
FE_kPCA = 0
FE_kPLS = 1
FE_LDA = 0
FE_ICA = 0
#==============================================================================
#                     LOADING DATA
#==============================================================================

"""
    Xtrain: Training sample vector matrix with dimensions (NsamplesTr, Ndim)
    Xtest: Testing sample vector with matrix dimensions (NsamplesTst, Ndim)
    Ytrain: Target Training output for the tr samples  (NsamplesTr, 1)
    Ytest: Target Testing output for the tst samples (NsamplesTst, 1)
"""


#==============================================================================
#==============================================================================
#==============================================================================
#                           Preprocess
#==============================================================================
#==============================================================================
#==============================================================================
if (PREPROCESS_F == 1):
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)            
    Xtest = scaler.transform(Xtest)
    
    # Create one-vs-all output matrix
    Ytrain_m = np.zeros((len(Ytrain),n_person))
    for i in range (n_person):
        Ytrain_m[np.where(Ytrain == i),i] = True


# In[6]:

#==============================================================================
#                           Feature Extraction
#==============================================================================

from sklearn.cross_decomposition import PLSSVD,PLSCanonical,PLSRegression
import sklearn.metrics.pairwise as pair
from sklearn.preprocessing import KernelCenterer
from sklearn.kernel_approximation import Nystroem,RBFSampler


n_comp = 15   # Number of features (components) to be obtained
n_class = 2   # Number of classes for the classification

#==============================================================================
# Linear PCA

if (FE_PCA == 1):
    from sklearn.decomposition import PCA 
    pca = PCA(n_components = n_comp, whiten = False)  # Whiten would preprocess data and make hard assumptions blabla
    pca.fit(Xtrain)     # Obtain the components from the data 
    
    PCA_hyperplanes = pca.components_  # Components of the descomposition (hyperplanes) (eigenvesctors) (eigenfaces) 
    PCA_S = pca.explained_variance_ratio_  # Percentage of variance that each component explains (eigenvectors)
   
    Xtrain = pca.fit_transform(Xtrain)  # It obtains the features of the components.PCA
    Xtest = pca.transform(Xtest)

# Kernel PCA

if (FE_kPCA == 1):
    from sklearn.decomposition import KernelPCA
    # Get proper value for the gamma of the gaussian projection
    d = pair.pairwise_distances(Xtrain,Xtrain)
    aux = np.triu(d)
    sigma = np.sqrt(np.mean(np.power(aux[aux!=0],2)*0.5))
    gamma = 1/(2*sigma**2)
    
    kpca = KernelPCA(n_components = n_comp,kernel = "rbf",gamma = gamma)
    kpca.fit(Xtrain)
    
    kPCA_hyperplanes = kpca.alphas_  # Components of the descomposition (hyperplanes) (eigenvesctors) (eigenfaces) 
    kPCA_S = kpca.lambdas_           # Percentage of variance that each component explains (eigenvectors)
   
    Xtrain = kpca.transform(Xtrain)
    Xtest = kpca.transform(Xtest)
    

# LDA

if (FE_LDA == 1):
    from sklearn.lda import LDA
    lda = LDA()
    lda.fit(Xtrain,Ytrain)
    
    LDA_centroids = lda.means_    # Centroids of the classes (n_class, n_features)

    Xtrain = lda.transform(Xtrain)
    Xtest = lda.transform(Xtest)
    
# Linear PLS

if (FE_PLS == 1):
    pls2 = PLSRegression(n_components=n_comp)
    pls2.fit(Xtrain,Ytrain_m)
    pls2
    Xtrain = pls2.transform(Xtrain)
    Xtest = pls2.transform(Xtest)
    

    
# Kernel PLS

if (FE_kPLS == 1):
    d = pair.pairwise_distances(Xtrain,Xtrain)
    aux = np.triu(d)
    sigma = np.sqrt(np.mean(np.power(aux[aux!=0],2)*0.5))
    gamma = 1/(2*sigma**2)
    
    ktrain = pair.rbf_kernel(Xtrain,Xtrain,gamma)
    ktest = pair.rbf_kernel(Xtest,Xtrain,gamma)
    kcent = KernelCenterer()
    kcent.fit(ktrain)
    ktrain = kcent.transform(ktrain)
    ktest = kcent.transform(ktest)
    
    kpls = PLSRegression(n_components = n_comp)
    kpls.fit(ktrain,Ytrain_m)
    
    Xtrain = kpls.transform(ktrain)
    Xtest = kpls.transform(ktest)
    
# Linear CCA  Cannonical Correlation Análisis

if (FE_CCA == 1):
    from sklearn.cross_decomposition import CCA
    cca = CCA(n_components = n_class)
    cca.fit(Xtrain,Ytrain_m)
    Xtrain = cca.transform(Xtrain)
    Xtest = cca.transform(Xtest)
    

# ICA Independen Component Analisis
if (FE_ICA == 1):
    from sklearn.decomposition import FastICA
    ica = FastICA(n_components = n_comp)
    ica.fit(Xtrain)   # Reconstruct signals
    ICA_components = ica.components_  # Get components 
    ICA_mixing = ica.mixing_  # Get estimated mixing matrix
    
    Xtrain = ica.transform(Xtrain)
    Xtest = ica.transform(Xtest)






