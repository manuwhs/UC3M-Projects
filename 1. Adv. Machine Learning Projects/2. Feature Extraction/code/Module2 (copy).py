import numpy as np
import matplotlib.pyplot as plt
import Utilities as util
import timeit

from sklearn.decomposition import PCA,KernelPCA
from sklearn.cross_decomposition import PLSSVD,PLSCanonical,PLSRegression
from sklearn.cross_decomposition import CCA
from sklearn.lda import LDA
import sklearn.metrics.pairwise as pair
from sklearn.preprocessing import KernelCenterer
from sklearn.kernel_approximation import Nystroem,RBFSampler

plt.close('all')

# Load data
data = np.loadtxt("AVIRIS_dataset/data.txt")
labels = np.loadtxt("AVIRIS_dataset/labels.txt")
names = np.loadtxt("AVIRIS_dataset/names.txt", dtype=np.str)

# Exclude background class
dataR = data[labels!=0,:]
labelsR = labels[labels!=0]
nClasses = np.alen(np.unique(labelsR))
nFeatures = np.shape(dataR)[1]

# Split data in training and test sets
rang = np.arange(np.shape(dataR)[0],dtype=int)
np.random.seed(0)
rang = np.random.permutation(rang)

Ntrain = round(0.2*np.shape(dataR)[0])
Ntest = len(rang)-Ntrain
dataTrain = dataR[rang[:Ntrain]]
dataTest = dataR[rang[Ntrain:]]
labelsTrain = labelsR[rang[:Ntrain]]
labelsTest = labelsR[rang[Ntrain:]]

# Normalize data
mx = np.mean(dataTrain,axis=0,dtype=np.float64)
stdx = np.std(dataTrain,axis=0,dtype=np.float64)
dataTrain = np.divide(dataTrain-np.tile(mx,[Ntrain,1]),np.tile(stdx,[Ntrain,1]))
dataTest = np.divide(dataTest-np.tile(mx,[Ntest,1]),np.tile(stdx,[Ntest,1]))

# Create output coding matrix Y
Ytrain = np.zeros((Ntrain,np.alen(np.unique(labelsTrain))))
for i in np.unique(labelsTrain):
    Ytrain[labelsTrain==i,i-1] = True
Ytest = np.zeros((Ntest,np.alen(np.unique(labelsTest))))
for i in np.unique(labelsTest):
    Ytest[labelsTest==i,i-1] = True
    
# Create a set of random colors for each class
np.random.seed()
classColors = np.random.rand(nClasses,3)

# Global variables
nComponents = np.arange(1,nFeatures,8)

### LINEAR METHODS ###
#%% PRINCIPAL COMPONENT ANALYSIS
# Plot explained variance vs number of components
pca = PCA()
pca.fit_transform(dataTrain)
cumvarPCA = np.cumsum(pca.explained_variance_ratio_)

# Plot classification score vs number of components
nComponents = np.arange(1,nFeatures,8)
pcaScores = np.zeros((2,np.alen(nComponents)))
for i,n in enumerate(nComponents):   
    pca = PCA(n_components=n,whiten=False)
    dataTrainT = pca.fit_transform(dataTrain)
    dataTestT = pca.transform(dataTest)
    pcaScores[:,i] = util.classify(dataTrainT,dataTestT,labelsTrain,labelsTest)

# Training data with 2 dimensions
pca = PCA(n_components=2)
xtPCA = pca.fit_transform(dataTrain)
uPCA = pca.components_

#%% PARTIAL LEAST SQUARES
#%% PLS SVD
nComponents = np.arange(1,nClasses+1)
plsSvdScores = np.zeros((2,np.alen(nComponents)))
for i,n in enumerate(nComponents):
    plssvd = PLSSVD(n_components=n)
    plssvd.fit(dataTrain,Ytrain)
    dataTrainT = plssvd.transform(dataTrain)
    dataTestT = plssvd.transform(dataTest)
    plsSvdScores[:,i] = util.classify(dataTrainT,dataTestT,labelsTrain,labelsTest)
fig = plt.figure()
util.plotAccuracy(fig,nComponents,plsSvdScores)
plt.title('PLS SVD accuracy',figure=fig)

plssvd = PLSSVD(n_components=2)
xt,yt = plssvd.fit_transform(dataTrain,Ytrain)
fig = plt.figure()
util.plotData(fig,xt,labelsTrain,classColors)

u = plssvd.x_weights_
plt.quiver(u[0,0],u[1,0],color='k',edgecolor='k',lw=1,scale=0.1,figure=fig)
plt.quiver(-u[1,0],u[0,0],color='k',edgecolor='k',lw=1,scale=0.4,figure=fig)

#%% PLS mode-A
lda = LDA()
nComponents = np.arange(1,nClasses+1)
plsCanScores = np.zeros((2,np.alen(nComponents)))
for i,n in enumerate(nComponents):
    plscan = PLSCanonical(n_components=n)
    plscan.fit(dataTrain,Ytrain)
    dataTrainT = plscan.transform(dataTrain)
    dataTestT = plscan.transform(dataTest)
    plsCanScores[:,i] = util.classify(dataTrainT,dataTestT,labelsTrain,labelsTest)
fig = plt.figure()
util.plotAccuracy(fig,nComponents,plsCanScores)
plt.title('PLS Canonical accuracy',figure=fig)

plscan = PLSCanonical(n_components=2)
xt,yt = plscan.fit_transform(dataTrain,Ytrain)
fig = plt.figure()
util.plotData(fig,xt,labelsTrain,classColors)

u = plscan.x_weights_
plt.quiver(u[0,0],u[1,0],color='k',edgecolor='k',lw=1,scale=0.1,figure=fig)
plt.quiver(-u[1,0],u[0,0],color='k',edgecolor='k',lw=1,scale=0.4,figure=fig)

#%% PLS2
lda = LDA()
nComponents = np.arange(1,nFeatures,8)
pls2Scores = np.zeros((2,np.alen(nComponents)))
for i,n in enumerate(nComponents):
    pls2 = PLSRegression(n_components=n)
    pls2.fit(dataTrain,Ytrain)
    dataTrainT = pls2.transform(dataTrain)
    dataTestT = pls2.transform(dataTest)
    pls2Scores[:,i] = util.classify(dataTrainT,dataTestT,labelsTrain,labelsTest)

pls2 = PLSRegression(n_components=2)
xtPLS,yt = pls2.fit_transform(dataTrain,Ytrain)

uPLS = pls2.x_weights_

#%% Canonical Correlation Analysis
nComponents = np.arange(1,nClasses+1)
cca = CCA(n_components=nClasses)
cca.fit(dataTrain,Ytrain)
dataTrainT = cca.transform(dataTrain)
dataTestT = cca.transform(dataTest)
ccaScores = np.zeros((2,np.alen(nComponents)))
for i,n in enumerate(nComponents):
    ccaScores[:,i] = util.classify(dataTrainT[:,0:n],dataTestT[:,0:n],labelsTrain,labelsTest)

#%% Linear Discriminant Analysis
nComponents = np.arange(1,nClasses+1)
ldaScores = np.zeros((2,np.alen(nComponents)))
for i,n in enumerate(nComponents):
    ldaT = LDA(n_components=n)
    ldaT.fit(dataTrain,labelsTrain)
    dataTrainT = ldaT.transform(dataTrain)
    dataTestT = ldaT.transform(dataTest)
    ldaScores[:,i] = util.classify(dataTrainT,dataTestT,labelsTrain,labelsTest)































#%% NONLINEAR METHODS %%#
d = pair.pairwise_distances(dataTrain,dataTrain)
aux = np.triu(d)
sigma = np.sqrt(np.mean(np.power(aux[aux!=0],2)*0.5))
gamma = 1/(2*sigma**2)

#%% K-PCA
# Calculate accumulated variance
kpca = KernelPCA(kernel="rbf",gamma=gamma)
kpca.fit_transform(dataTrain)
eigenvals = kpca.lambdas_[0:220]
sumeig = np.sum(eigenvals)
var = np.divide(eigenvals,sumeig)
cumvarkPCA = np.cumsum(var)

# Calculate classifiation scores for each component
nComponents = np.arange(1,nFeatures,8)
kpcaScores = np.zeros((2,np.alen(nComponents)))
for i,n in enumerate(nComponents):   
    kpca = KernelPCA(n_components=n,kernel="rbf",gamma=gamma)
    kpca.fit(dataTrain)
    dataTrainT = kpca.transform(dataTrain)
    dataTestT = kpca.transform(dataTest)
    kpcaScores[:,i] = util.classify(dataTrainT,dataTestT,labelsTrain,labelsTest)

# K-PCA second round
ktrain = pair.rbf_kernel(dataTrain,dataTrain,gamma)
ktest = pair.rbf_kernel(dataTest,dataTrain,gamma)
kcent = KernelCenterer()
kcent.fit(ktrain)
ktrain = kcent.transform(ktrain)
ktest = kcent.transform(ktest)

kpca = PCA()
kpca.fit_transform(ktrain)
cumvarkPCA2 = np.cumsum(kpca.explained_variance_ratio_[0:220])

# Calculate classifiation scores for each component
nComponents = np.arange(1,nFeatures,8)
kpcaScores2 = np.zeros((2,np.alen(nComponents)))
for i,n in enumerate(nComponents):   
    kpca2 = PCA(n_components=n)
    kpca2.fit(ktrain)
    dataTrainT = kpca2.transform(ktrain)
    dataTestT = kpca2.transform(ktest)
    kpcaScores2[:,i] = util.classify(dataTrainT,dataTestT,labelsTrain,labelsTest)

#%% K-PLS
ktrain = pair.rbf_kernel(dataTrain,dataTrain,gamma)
ktest = pair.rbf_kernel(dataTest,dataTrain,gamma)
kcent = KernelCenterer()
kcent.fit(ktrain)
ktrain = kcent.transform(ktrain)
ktest = kcent.transform(ktest)

nComponents = np.arange(1,nFeatures,8)
kplsScores = np.zeros((2,np.alen(nComponents)))
for i,n in enumerate(nComponents):   
    kpls = PLSRegression(n_components=n)
    kpls.fit(ktrain,Ytrain)
    dataTrainT = kpls.transform(ktrain)
    dataTestT = kpls.transform(ktest)
    kplsScores[:,i] = util.classify(dataTrainT,dataTestT,labelsTrain,labelsTest)

kpls = PLSRegression(n_components=50)
startTime = timeit.default_timer()
kpls.fit(ktrain,Ytrain)
elapTime = timeit.default_timer() - startTime
dataTrainT = kpls.transform(ktrain)
dataTestT = kpls.transform(ktest)
kplsScoreFull = util.classify(dataTrainT,dataTestT,labelsTrain,labelsTest)

#%% K-CCA
ktrain = pair.rbf_kernel(dataTrain,dataTrain,gamma)
ktest = pair.rbf_kernel(dataTest,dataTrain,gamma)
kcent = KernelCenterer()
kcent.fit(ktrain)
ktrain = kcent.transform(ktrain)
ktest = kcent.transform(ktest)

nComponents = np.arange(1,nClasses+1)
kcca = CCA(n_components=nClasses)
kcca.fit(ktrain,Ytrain)
dataTrainT = kcca.transform(ktrain)
dataTestT = kcca.transform(ktest)
kccaScores = np.zeros((2,np.alen(nComponents)))
for i,n in enumerate(nComponents):   
    kccaScores[:,i] = util.classify(dataTrainT[:,0:n],dataTestT[:,0:n],labelsTrain,labelsTest)

#%% Subsampling methods
kpls = PLSRegression(n_components=150)
nComponents = np.arange(173,2173,100)

# Nystroem method
elapTimeNys = np.zeros(np.shape(nComponents))
kplsScoresNys = np.zeros((2,3))
for i,n in enumerate(nComponents):
    nys = Nystroem(n_components=n,gamma=gamma)
    nys.fit(dataTrain)
    ktrain = nys.transform(dataTrain)
    ktest = nys.transform(dataTest)
    startTime = timeit.default_timer()
    kpls.fit(ktrain,Ytrain)
    elapTimeNys[i] = timeit.default_timer() - startTime
    dataTrainT = kpls.transform(ktrain)
    dataTestT = kpls.transform(ktest)
    
    if n==573:
        kplsScoresNys[:,0] = util.classify(dataTrainT,dataTestT,labelsTrain,labelsTest)
    elif n==1073:
        kplsScoresNys[:,1] = util.classify(dataTrainT,dataTestT,labelsTrain,labelsTest)
    elif n==1573:
        kplsScoresNys[:,2] = util.classify(dataTrainT,dataTestT,labelsTrain,labelsTest)

# RBF sampler method
elapTimeRBFS = np.zeros(np.shape(nComponents))
kplsScoresRBFS = np.zeros((2,3))
for i,n in enumerate(nComponents):
    rbfs = RBFSampler(n_components=n,gamma=gamma)
    rbfs.fit(dataTrain)
    ktrain = rbfs.transform(dataTrain)
    ktest = rbfs.transform(dataTest)
    startTime = timeit.default_timer()
    kpls.fit(ktrain,Ytrain)
    elapTimeRBFS[i] = timeit.default_timer() - startTime
    dataTrainT = kpls.transform(ktrain)
    dataTestT = kpls.transform(ktest)
    
    if n==573:
        kplsScoresRBFS[:,0] = util.classify(dataTrainT,dataTestT,labelsTrain,labelsTest)
    elif n==1073:
        kplsScoresRBFS[:,1] = util.classify(dataTrainT,dataTestT,labelsTrain,labelsTest)
    elif n==1573:
        kplsScoresRBFS[:,2] = util.classify(dataTrainT,dataTestT,labelsTrain,labelsTest)
        
#%% Plot figures
#%% Plot explained variance vs number of components for PCA and kPCA
plt.figure()
plt.plot(np.arange(1,np.alen(cumvarPCA)+1),cumvarPCA,c='c',lw=2,label='Linear PCA')
plt.plot(np.arange(1,np.alen(cumvarkPCA2)+1),cumvarkPCA2,c='r',lw=2,label='Gaussian kernel PCA')
plt.xlim(1,np.alen(cumvarPCA))
plt.legend(loc='lower right')
plt.title('Explained Variance Ratio')
plt.xlabel('number of components')
plt.ylabel('explained variance ratio')
plt.grid(True)
plt.show()

#%% Plot accuracies for PCA vs kPCA
nComponents = np.arange(1,nFeatures,8)
plt.figure()
plt.plot(nComponents,pcaScores[0,:],'c',lw=2,label='Linear SVM for PCA')
plt.plot(nComponents,pcaScores[1,:],'r',lw=2,label='RBF SVM for PCA')
plt.plot(nComponents,kpcaScores2[0,:],'c--',lw=2,label='Linear SVM for kernelPCA')
plt.plot(nComponents,kpcaScores2[1,:],'r--',lw=2,label='RBF SVM for kernelPCA')
plt.xlim(1,np.amax(nComponents))
plt.title('PCA and kernelPCA classification performance')
plt.xlabel('number of components')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.grid(True)

#%% Plot data proyections for PCA
labels = np.unique(labelsTrain.astype(np.int))
plt.figure()
for i,l in enumerate(labels):
    plt.scatter(xtPCA[labelsTrain==l,0],xtPCA[labelsTrain==l,1],alpha=0.5,c=classColors[i,:])
plt.quiver(uPCA[0,0],uPCA[1,0],color='k',edgecolor='k',lw=1,scale=0.2)
plt.quiver(-uPCA[1,0],uPCA[0,0],color='k',edgecolor='k',lw=1,scale=0.6)
plt.title('2 dimensional PCA training data')
plt.show()

#%% Plot data proyections for PLS2
plt.figure()
for i,l in enumerate(labels):
    plt.scatter(xtPLS[labelsTrain==l,0],xtPLS[labelsTrain==l,1],alpha=0.5,c=classColors[i,:])
plt.quiver(uPLS[0,0],uPLS[1,0],color='k',edgecolor='k',lw=1,scale=0.3)
plt.quiver(-uPLS[1,0],uPLS[0,0],color='k',edgecolor='k',lw=1,scale=0.6)
plt.title('2 dimensional PLS2 training data')
plt.show()

#%% Plot accuracies for linear methods
nComponents = np.arange(1,nFeatures,8)
nComponents2 = np.arange(1,nClasses+1)
plt.figure()
plt.plot(nComponents,pcaScores[0,:],'c',lw=2,label='Linear SVM for PCA')
plt.plot(nComponents,pls2Scores[0,:],'r',lw=2,label='Linear SVM for PLS2')
plt.plot(nComponents2,ccaScores[0,:],'y',lw=2,label='Linear SVM for CCA')
plt.plot(nComponents2,ldaScores[0,:],'k',lw=2,label='Linear SVM for LDA')
plt.xlim(1,np.amax(nComponents))
plt.title('Comparison between linear methods')
plt.xlabel('number of components')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.grid(True)

#%% Plot accuracies for nonlinear methods
nComponents = np.arange(1,nFeatures,8)
plt.figure()
plt.plot(nComponents,kpcaScores2[0,:],'c',lw=2,label='Linear SVM for PCA')
plt.plot(nComponents,kplsScores[0,:],'r',lw=2,label='Linear SVM for PLS2')
#plt.plot(nComponents,kccaScores[0,:],'y',lw=2,label='Linear SVM for CCA')
plt.xlim(1,np.amax(nComponents))
plt.title('Comparison between nonlinear methods')
plt.xlabel('number of components')
plt.ylabel('accuracy')
plt.legend(loc='upper left')
plt.grid(True)

#%% Plot execution time for kernel PLS2 and subsampled methods
nComponents = np.arange(173,2173,100)
plt.figure()
plt.plot(nComponents,np.tile(elapTime,(20)),'k',lw=2,label='Full kernel')
plt.plot(nComponents,elapTimeNys,'r',lw=2,label='Nystroem method')
plt.plot(nComponents,elapTimeRBFS,'y',lw=2,label='RBF sampler')
plt.xlim(173,np.amax(nComponents))
plt.title('Execution time comparison for full and compacted kernels')
plt.xlabel('number of kernel components')
plt.ylabel('time units')
plt.legend(loc='upper left')
plt.grid(True)

#%% Plot accuracies for some number of components with linear SVM
plt.figure()
plt.bar(2,kplsScoreFull[0],width=0.4,color='k',align='center')
plt.bar(np.arange(4,10,2),kplsScoresNys[0,:][::-1],width=0.5,color='c',align='center',label='Nystroem method')
plt.bar(np.arange(4,10,2)+0.5,kplsScoresRBFS[0,:][::-1],width=0.5,color='y',align='center',label='RBF sampler')
plt.xticks([2,4.25,6.25,8.25],['Full kernel','1573','1073','573'])
plt.xlabel('number of kernel components')
plt.ylabel('accuracies')
plt.title('Accuracies for several compacted kernel matrices')
plt.legend(loc='lower right')
plt.show()
