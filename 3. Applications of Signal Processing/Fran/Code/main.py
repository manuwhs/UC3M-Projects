import matplotlib.pyplot as plt
import img_util as imgu
import numpy as np
import scipy.io       # To load Matlab files


plt.close('all')

#######################################################################
##                           Load images 
#######################################################################

# Matlab data files are like dictionaries !!
data = scipy.io.loadmat('Yale_64x64.mat')
img_matrix = data['fea']  # MATRIX WITH ALL THE IMAGES FLATTENED
person = data['gnd']  # Array with the label of the people.

num_img,dim = img_matrix.shape               # Preprocess images
for i in range (num_img): 
    img = img_matrix[i].reshape((64,64)).T  # Dunno why images were transposed
    img_matrix[i] = img.flatten()
    
m = 64    # Number of columns
n = 64   


X = img_matrix;
Y = person.ravel();
Y = Y - 1;   # Just in case we make the classes start in 0, (0 - 14)

# Split the dataset into equeall
from sklearn.cross_validation import StratifiedShuffleSplit
SSS =  StratifiedShuffleSplit(Y, 1, test_size = 0.2, random_state = 0)
for train_index, test_index in SSS:
#    print "TRAIN:", train_index, "TEST:", test_index
    Xtrain, Xtest = X[train_index], X[test_index]
    Ytrain, Ytest = Y[train_index], Y[test_index]

Ntrain, Ndim = Xtrain.shape
Ntest, Ndim = Xtest.shape

n_person = Ytrain[-1]

#######################################################################
##                           DATA PREPROCESS
#######################################################################

PREPROCESS_F = 1;

if (PREPROCESS_F == 1):
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)            
    Xtest = scaler.transform(Xtest)
    
# Create one-vs-all output matrix
Ytrain_m = np.zeros((len(Ytrain),n_person))
for i in range (n_person):
    Ytrain_m[np.where(Ytrain == i+1),i] = True
    
READ_SHOW_IMG = 0


HISTOGRAM_EQ = 1;

if ( HISTOGRAM_EQ == 1):
    pers = 35
    nbr_bins = 255;
    plt.figure()
    
    plt.subplot(2,2,1)
    plt.imshow(Xtrain[pers,:].reshape(m,n))
    plt.subplot(2,2,2)
    imhist,bins = np.histogram(Xtrain[pers,:].flatten(),nbr_bins,normed=True)
    plt.bar(bins[:255],imhist,width = 0.1 ,color='c',align='center')
    
    # Equalization is 
    for i in range (Ntrain):
        Xtrain[i,:] = imgu.histeq(Xtrain[i,:]);
        
    for i in range (Ntest):
        Xtest[i,:] = imgu.histeq(Xtest[i,:]);

    plt.subplot(2,2,3)
    plt.imshow(Xtrain[pers,:].reshape(m,n))
    plt.subplot(2,2,4)
    imhist,bins = np.histogram(Xtrain[pers,:].flatten(),nbr_bins,normed=True)
    plt.bar(bins[:255],imhist,width =0.1,color='c',align='center')

 
    
if (READ_SHOW_IMG == 1):
    # Show 1 image from the Dataset
    plt.figure()
    for i in range(8):
        plt.subplot(2,4,i)
        plt.imshow(X[i*13].reshape(m,n))

# perform PCA.
""" 
PCA will output as many eigenfaces (planes) as training samples, 
to obtian the feature associated to a eigenface, proyect the image into 
the eigenface plane. (No bias ?? Could we add one ?)

"""
#######################################################################
#######################################################################
##                           FEATURE EXTRACTION
#######################################################################
#######################################################################

FE_PCA = 1;
FE_kPCA = 0;
FE_ICA = 0;
FE_LDA = 0;
FE_PLS = 0;

# Linear PCA
if (FE_PCA == 1):
    n_comp = 100
    
#    Xtrain -= np.mean(Xtrain, axis = 0)
    from sklearn.decomposition import PCA 
    pca = PCA(n_components = n_comp, whiten = False)  # Whiten would preprocess data and make hard assumptions blabla
    pca.fit(Xtrain)     # Obtain the components from the data 
    
    PCA_hyperplanes = pca.components_  # Components of the descomposition (hyperplanes) (eigenvesctors) (eigenfaces) 
    PCA_S = pca.explained_variance_ratio_  # Percentage of variance that each component explains (eigenvectors)
    PCA_mean = pca.mean_
    Xtrain = pca.fit_transform(Xtrain)  # It obtains the features of the components.PCA
    Xtest = pca.transform(Xtest)
    
    
    # Plot the reconstruction of a face !! 
#==============================================================================
#     plt.figure()
#     plt.gray()
#     comp = [2, 8, 20, 40, 60, 80, 100, 120]
#     aux = 1
#     for k in comp:
#         img = PCA_mean[:].copy()
#         for i in range (k):
#             img += Xtrain[0,i]*PCA_hyperplanes[i,:]
#             
#         ax = plt.subplot(2,4,aux)
#         ax.set_title(str(k),fontsize = 20)
#         plt.imshow(img.reshape(m,n))
#     aux += 1
# 
#==============================================================================
    
#==============================================================================
#     # show so me images (mean and 7 first modes)
#     plt.figure()
#     plt.gray()
#     
#     plt.title("Mean face", fontsize = 40, y = 1.01)
#     plt.imshow(PCA_mean.reshape(m,n))
#     
#     plt.figure()
#     plt.gray()
#     plt.suptitle("PCA eigenfaces")
#     for i in range(8):
#         plt.subplot(2,4,i)
#         plt.imshow(PCA_hyperplanes[i].reshape(m,n))
#==============================================================================


        
# Kernel PCA
if (FE_kPCA == 1):
    from sklearn.decomposition import KernelPCA
    import sklearn.metrics.pairwise as pair
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
    
# ICA Independen Component Analisis
if (FE_ICA == 1):
    from sklearn.decomposition import FastICA
    ica = FastICA(n_components = 15, max_iter = 10000,  tol=0.00001 )
#    Xtrain -= np.mean(Xtrain, axis = 0)
    
    ica.fit(Xtrain)   # Reconstruct signals
    ICA_components = ica.components_  # Get components 
    ICA_mixing = ica.mixing_  # Get estimated mixing matrix
    
    Xtrain = ica.transform(Xtrain)
    Xtest = ica.transform(Xtest)

    plt.figure()
    plt.gray()
    plt.suptitle("ICA faces 15 components", fontsize = 40, y = 1.0)
    for i in range(8):
        plt.subplot(2,4,i)
        plt.imshow(ICA_components[i].reshape(m,n))
        
# LDA
if (FE_LDA == 1):
    from sklearn.lda import LDA
    lda = LDA()
    lda.fit(Xtrain,Ytrain.ravel())
    
    LDA_centroids = lda.means_    # Centroids of the classes (n_class, n_features)
    LDA_hyperplanes = lda.coef_
    
    Xtrain = lda.transform(Xtrain)
    Xtest = lda.transform(Xtest)
    
    plt.figure()
    plt.gray()
    plt.suptitle("LDA centroids", fontsize = 40, y = 1.0)
    for i in range(8):
        plt.subplot(2,4,i)
        plt.imshow(LDA_centroids[i].reshape(m,n))
        
    plt.figure()
    plt.gray()
    plt.suptitle("LDA hyperplanes", fontsize = 40, y = 1.0)
    for i in range(8):
        plt.subplot(2,4,i)
        plt.imshow(LDA_hyperplanes[i].reshape(m,n))

# PLS 
if (FE_PLS == 1):
    from sklearn.cross_decomposition import PLSSVD,PLSCanonical,PLSRegression
    pls = PLSRegression(n_components = 15)

#    Xtrain -= np.mean(Xtrain, axis = 0)

    pls.fit(Xtrain,Ytrain_m)
    
    PLS_weights = pls.x_weights_.T
    
    Xtrain = pls.transform(Xtrain)
    Xtest = pls.transform(Xtest)
    
    plt.figure()
    plt.gray()
    
    plt.suptitle("PLS hyperplanes", fontsize = 40, y = 1.0)
    for i in range(8):
        plt.subplot(2,4,i)
        plt.imshow(PLS_weights[i].reshape(m,n))
        
#######################################################################
#######################################################################
##                           FEATURE SELECTION
#######################################################################
#######################################################################

FS_Random_Forest = 0 # Random forest feature selection
FS_LSVC = 0
FS_RFE = 0
FS_RFECV = 0
FS_Random_Forest = 0


from sklearn.ensemble import ExtraTreesClassifier
from sklearn import random_projection
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
max_features = 10
    
#==============================================================================
# Random Forest

if (FS_Random_Forest == 1):
    print "Performing RF_FS"

    et = ExtraTreesClassifier(n_estimators=1000,criterion='entropy',n_jobs=-1)
    et.fit(Xtrain,Ytrain)
    rf_scores = et.feature_importances_
    rf_index = np.argsort(rf_scores)[::-1]
    rf_scores = rf_scores[rf_index]
    rf_scores = rf_scores/np.max(rf_scores)
    plt.figure()
    plt.scatter(rf_index,rf_scores,color="blue")
    
#    i = 0
#    tf_RF = 0.01;  # Threshold of relative importance of the characteristics.
#    selected_feature_indx = []
#    while (rf_scores[i] > th_RF):
#        selected_feature_indx.append(rf_index[i])
#        i += 1
#
#    Xtrain_sel = []
#    Xtest_sel = []
#    
#    for indx in selected_feature_indx:
#        Xtrain_sel.append(Xtrain[:,indx])
#        Xtest_sel.append(Xtest[:,indx])
#        
#    Xtrain = np.array(Xtrain_sel).T
#    Xtest = np.array(Xtest_sel).T

#==============================================================================
# Linear SVC 

if (FS_LSVC == 1):
    print "Performing LSVC_FS"
    from sklearn.svm import LinearSVC
    
    LSVC =  LinearSVC(C=0.1, penalty="l1", dual=False)
    selected_Xtrain = LSVC.fit_transform(Xtrain, Ytrain)
    selected_Xtest = LSVC.transform(Xtest)
    Xtrain = selected_Xtrain
    Xtest = selected_Xtest


# Recursive Feature Elimination

if (FS_RFE == 1):
    print "Performing RFE_FS"
    est = SVC(kernel='linear')
    rfe = RFE(est)
    rfe.fit(Xtrain,Ytrain)
    rfe_scores = 1/rfe.ranking_.astype(np.float)
    rfe_index = np.argsort(rfe_scores)[::-1]
    rfe_scores = rfe_scores[rfe_index]
    rfe_scores = rfe_scores/np.max(rfe_scores)
    print('RFE Train Accuracy: '+str(rfe.score(Xtrain,Ytrain)))

    pl.figure()
    pl.scatter(rfe_index,rfe_scores,color="blue")

    Xtrain_sel = []
    Xtest_sel = []
    
    for i in range(len(rfe_scores)):
        if (rfe_scores[i] == 1):
            Xtrain_sel.append(Xtrain[:,i])
            Xtest_sel.append(Xtest[:,i])
        
    Xtrain = np.array(Xtrain_sel).T
    Xtest = np.array(Xtest_sel).T
    
# Recursive Feature Elimination Cross Validation using SVM

if (FS_RFECV == 1):
    print "Performing RFECV_FS"
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.feature_selection import RFECV

    est = SVC(kernel='linear')
    
    stkfold = StratifiedKFold(Ytrain,n_folds=5)
    
    rfecv = RFECV(estimator=est, step=1, cv=stkfold,
              scoring='accuracy')

    rfecv.fit(Xtrain,Ytrain)
    
    rfecv_scores = 1/rfecv.ranking_.astype(np.float)
    rfecv_index = np.argsort(rfecv_scores)[::-1]
    rfecv_scores = rfecv_scores[rfecv_index]
    rfecv_scores = rfecv_scores/np.max(rfecv_scores)
    print('RFECV Train Accuracy: '+str(rfecv.score(Xtrain,Ytrain)))

    pl.figure()
    pl.scatter(rfecv_index,rfecv_scores,color="blue")
    
    Xtrain_sel = []
    Xtest_sel = []
    
    for i in range(len(rfecv_scores)):
        if (rfecv_scores[i] == 1):
            Xtrain_sel.append(Xtrain[:,i])
            Xtest_sel.append(Xtest[:,i])
        
    Xtrain = np.array(Xtrain_sel).T
    Xtest = np.array(Xtest_sel).T


#######################################################################
#######################################################################
##                           CLASSIFICATION
#######################################################################
#######################################################################
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,BaggingClassifier,AdaBoostClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import  make_scorer     # To make a scorer for the GridSearch.

scaler = preprocessing.StandardScaler().fit(Xtrain)
Xtrain = scaler.transform(Xtrain)            
Xtest = scaler.transform(Xtest)
    
LR_cl = 1;
SVM_cl = 00;
LDA_cl = 00;

if (LR_cl == 1):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(Xtrain,Ytrain)
    scores = np.empty((4))
    scores[0] = lr.score(Xtrain,Ytrain)
    scores[1] = lr.score(Xtest,Ytest)
    print('Logistic Regression, train: {0:.02f}% '.format(scores[0]*100))
    print('Logistic Regression, test: {0:.02f}% '.format(scores[1]*100))
    
    # Bagging sklearn
    blr = BaggingClassifier(LogisticRegression(),n_estimators = 100, max_samples = 0.9,n_jobs=-1)
    blr.fit(Xtrain,Ytrain)
    scores[2] = blr.score(Xtrain,Ytrain)
    scores[3] = blr.score(Xtest,Ytest)
    print('Bagging Logistic Regression, train: {0:.02f}% '.format(scores[2]*100))
    print('Bagging Logistic Regression, test: {0:.02f}% '.format(scores[3]*100))
    
    # MY  BAGGING
    import manu_ml as manl 
#    
#    blr = manl.Bagging_manu(LogisticRegression(),n_estimators = 100, max_samples = 0.8)
#    blr.fit(Xtrain,Ytrain)
#    scores[2] = blr.score(Xtrain,Ytrain)
#    scores[3] = blr.score(Xtest,Ytest)
#    print('Bagging Logistic Regression, train: {0:.02f}% '.format(scores[2]*100))
#    print('Bagging Logistic Regression, test: {0:.02f}% '.format(scores[3]*100))
    
    # Printing
    alg = ['LR','Bagged LR']
    trainsc = [scores[0],scores[2]]
    testsc = [scores[1],scores[3]]
    plt.figure()
    
    low = min(scores)
    high = max(scores)
    plt.ylim([0, 1.1])
    
    plt.bar(np.arange(2)+0.2,trainsc,width=0.4,color='#230439',align='center')
    plt.bar(np.arange(2)+0.6,testsc,width=0.4,color='#E70000',align='center')
    
    plt.xticks(np.arange(2)+0.4,alg, fontsize = 20)
    plt.title('Logistic Regression accuracy', fontsize = 20)
    plt.ylabel('Accuracy', fontsize = 20)
    plt.legend(['Train','Test'], fontsize = 20)
    plt.show()

#%% SVM Classifier
# Params C, kernel, degree, params of kernel

if (SVM_cl == 1):
    from sklearn.svm import SVC
    # Parameters for the validation
    C = np.logspace(-3,3,10)
    p = np.arange(2,5)
    gamma = np.array([0.125,0.25,0.5,1,2,4])/200
    
    # Create dictionaries with the Variables for the validation !
    # We create the dictinary for every TYPE of SVM we are gonna use.
    param_grid_linear = dict()
    param_grid_linear.update({'kernel':['linear']})
    param_grid_linear.update({'C':C})
    
    param_grid_pol = dict()
    param_grid_pol.update({'kernel':['poly']})
    param_grid_pol.update({'C':C})
    param_grid_pol.update({'degree':p})
    
    param_grid_rbf = dict()
    param_grid_rbf.update({'kernel':['rbf']})
    param_grid_rbf.update({'C':C})
    param_grid_rbf.update({'gamma':gamma})
    
    
    param = [{'kernel':'linear','C':C}]
    param_grid = [param_grid_linear,param_grid_pol,param_grid_rbf]
    
    # Validation is useful for validating a parameter, it uses a subset of the 
    # training set as "test" in order to know how good the generalization is.
    # The folds of "StratifiedKFold" are made by preserving the percentage of samples for each class.
    stkfold = StratifiedKFold(Ytrain, n_folds = 5)
    
    # The score function is the one we want to minimize or maximize given the label and the predicted.
    acc_scorer = make_scorer(accuracy_score)

    # GridSearchCV implements a CV over a variety of Parameter values !! 
    # In this case, over C fo the linear case, C and "degree" for the poly case
    # and C and "gamma" for the rbf case. 
    # The parameters we have to give it are:
    # 1-> Classifier Object: SVM, LR, RBF... or any other one with methods .fit and .predict
    # 2 -> Subset of parameters to validate. C 
    # 3 -> Type of validation: K-fold
    # 4 -> Scoring function. sklearn.metrics.accuracy_score

    gsvml = GridSearchCV(SVC(class_weight='auto'),param_grid_linear, scoring = acc_scorer,cv = stkfold, refit = True,n_jobs=-1)
    gsvmp = GridSearchCV(SVC(class_weight='auto'),param_grid_pol, scoring = acc_scorer,cv = stkfold, refit = True,n_jobs=-1)
    gsvmr = GridSearchCV(SVC(class_weight='auto'),param_grid_rbf, scoring =acc_scorer,cv = stkfold, refit = True,n_jobs=-1)
    
    gsvml.fit(Xtrain,Ytrain)
    gsvmp.fit(Xtrain,Ytrain)
    gsvmr.fit(Xtrain,Ytrain)
    
    trainscores = [gsvml.score(Xtrain,Ytrain),gsvmp.score(Xtrain,Ytrain),gsvmr.score(Xtrain,Ytrain)]
    testscores = [gsvml.score(Xtest,Ytest),gsvmp.score(Xtest,Ytest),gsvmr.score(Xtest,Ytest)]
    maxtrain = np.amax(trainscores)
    maxtest = np.amax(testscores)
    print('SVM, train: {0:.02f}% '.format(maxtrain*100))
    print('SVM, test: {0:.02f}% '.format(maxtest*100))
    
    #%% Bagging with SVM
    bsvml = BaggingClassifier(gsvml.best_estimator_,n_estimators=1000,max_samples= 0.90, n_jobs=-1, bootstrap = False)
    bsvmp = BaggingClassifier(gsvmp.best_estimator_,n_estimators=1000,max_samples= 0.90, n_jobs=-1, bootstrap = False)
    bsvmr = BaggingClassifier(gsvmr.best_estimator_,n_estimators=1000,max_samples= 0.90, n_jobs=-1, bootstrap = False)

    bsvml.fit(Xtrain,Ytrain)
    bsvmp.fit(Xtrain,Ytrain)
    bsvmr.fit(Xtrain,Ytrain)
    
    # Bagging Manu
#==============================================================================
#     import manu_ml as manl 
#     bsvml = manl.Bagging_manu(gsvml.best_estimator_,n_estimators=100,max_samples= 0.7)
#     bsvmp = manl.Bagging_manu(gsvmp.best_estimator_,n_estimators=100,max_samples= 0.7)
#     bsvmr = manl.Bagging_manu(gsvmr.best_estimator_,n_estimators=100,max_samples= 0.7)
# 
#     bsvml.fit(Xtrain,Ytrain)
#     bsvmp.fit(Xtrain,Ytrain)
#     bsvmr.fit(Xtrain,Ytrain)
#==============================================================================
    
    
    btrainscores = [bsvml.score(Xtrain,Ytrain),bsvmp.score(Xtrain,Ytrain),bsvmr.score(Xtrain,Ytrain)]
    maxtrain = np.amax(btrainscores)
    btestscores = [bsvml.score(Xtest,Ytest),bsvmp.score(Xtest,Ytest),bsvmr.score(Xtest,Ytest)]
    maxtest = np.amax(btestscores)
    print('Bagged SVM, train: {0:.02f}% '.format(maxtrain*100))
    print('Bagged SVM, test: {0:.02f}% '.format(maxtest*100))
    
    # Plot figure !!
    kernels = ['Linear\nC=2.15','Polynomic\n C=215.44\n degree=2','Gaussian\n C=215.44\n gamma=0.002']
    plt.figure()
    plt.subplot(1,2,1)
    plt.bar(np.arange(3)+0.2,trainscores,width=0.2,color='#230439',align='center')
    plt.bar(np.arange(3)+0.4,testscores,width=0.2,color='#E70000',align='center')
    plt.xticks(np.arange(3)+0.3,kernels, fontsize = 20)
    plt.title('SVM Accuracy', fontsize = 20)
    plt.ylabel('Accuracy', fontsize = 20)
    plt.legend(['Train','Test'], fontsize = 20)
    plt.subplot(1,2,2)
    plt.bar(np.arange(3)+0.2,btrainscores,width=0.2,color='#230439',align='center')
    plt.bar(np.arange(3)+0.4,btestscores,width=0.2,color='#E70000',align='center')
    plt.xticks(np.arange(3)+0.3,kernels, fontsize = 20)
    plt.title('Bagged SVM Accuracy', fontsize = 20)
    plt.ylabel('Accuracy', fontsize = 20)
    plt.legend(['Train','Test'], fontsize = 20)
    plt.show()

#%% LDA Classification

if (LDA_cl == 1):
    from sklearn.lda import LDA
    lda = LDA()
    lda.fit(Xtrain,Ytrain)
    scores = np.empty((4))
    scores[0] = lda.score(Xtrain,Ytrain)
    scores[1] = lda.score(Xtest,Ytest)
    print('LDA, train: {0:.02f}% '.format(scores[0]*100))
    print('LDA, test: {0:.02f}% '.format(scores[1]*100))
    
    blda = BaggingClassifier(LDA(),n_estimators = 10, max_samples = 0.95,n_jobs=-1, bootstrap = False,max_features = 1.0, bootstrap_features = False)
    blda.fit(Xtrain,Ytrain)
    scores[2] = blda.score(Xtrain,Ytrain)
    scores[3] = blda.score(Xtest,Ytest)
    print('Bagging LDA, train: {0:.02f}% '.format(scores[2]*100))
    print('Bagging LDA, test: {0:.02f}% '.format(scores[3]*100))
    
    # PLOT BARS 
    alg = ['LDA','Bagged LDA']
    trainsc = [scores[0],scores[2]]
    testsc = [scores[1],scores[3]]
    plt.figure()
    plt.bar(np.arange(2)+0.2,trainsc,width=0.4,color='#230439',align='center')
    plt.bar(np.arange(2)+0.6,testsc,width=0.4,color='#E70000',align='center')
    plt.xticks(np.arange(2)+0.4,alg, fontsize = 20)
    plt.title('Linear Discriminant Analysis accuracy', fontsize = 20)
    plt.ylabel('Accuracy', fontsize = 20)
    plt.legend(['Train','Test'], fontsize = 20)
    plt.show()
    
    