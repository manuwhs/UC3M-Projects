import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,BaggingClassifier,AdaBoostClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import  make_scorer     # To make a scorer for the GridSearch.

plt.close('all')

Load_Data = 1;
Data_Prep = 1;

LR_cl = 1;
LDA_cl = 0;
GNB_cl = 0;
KNN_cl = 0;
BT_cl = 0;
Tree_cl = 0;
SVM_cl = 0;
#%% Load data

if (Load_Data == 1):
    data = np.loadtxt("AVIRIS_dataset/data.txt")
    labels = np.loadtxt("AVIRIS_dataset/labels.txt")
    names = np.loadtxt("AVIRIS_dataset/names.txt", dtype=np.str)

#     Plot the data
#    plt.figure()
#    
#    ax = plt.subplot(1,4,1)   # First band for every pixel
#    img = data[:,1].reshape(145,145)
#    plt.imshow(img)
#    ax.set_title("Band 1",fontsize = 20)
#    
#    ax = plt.subplot(1,4,2)   # First band for every pixel
#    img = data[:,50].reshape(145,145)
#    plt.imshow(img)
#    ax.set_title("Band 50",fontsize = 20)
#
#    ax = plt.subplot(1,4,3)   # First band for every pixel
#    img = data[:,100].reshape(145,145)
#    plt.imshow(img)
#    ax.set_title("Band 100",fontsize = 20)
#
#    ax = plt.subplot(1,4,4)   # First band for every pixel
#    lab = labels.reshape(145,145)
#    plt.imshow(lab)
#    ax.set_title("Ground Truth",fontsize = 20)

#################################################################
#################### DATA PREPROCESSING #########################
#################################################################
if (Data_Prep == 1):
    #%% Remove noisy bands
    dataR1 = data[:,:103]
    dataR2 = data[:,108:149]
    dataR3 = data[:,163:219]
    dataR = np.concatenate((dataR1,dataR2,dataR3),axis=1)
    
    #%% Exclude background class
    dataR = dataR[labels!=0,:]
    labelsR = labels[labels!=0]
    labelsR = labelsR - 1  # So that classes start at 1
    #%% Split data in training and test sets
    train_ratio = 0.2
    rang = np.arange(np.shape(dataR)[0],dtype=int) # Create array of index
    np.random.seed(0)
    rang = np.random.permutation(rang)        # Randomize the array of index
    
    Ntrain = round(train_ratio*np.shape(dataR)[0])    # Number of samples used for training
    Ntest = len(rang)-Ntrain                  # Number of samples used for testing
    Xtrain = dataR[rang[:Ntrain]]
    Xtest = dataR[rang[Ntrain:]]
    Ytrain = labelsR[rang[:Ntrain]]
    Ytest = labelsR[rang[Ntrain:]]
    
    #%% Normalize data
    mx = np.mean(Xtrain,axis=0,dtype=np.float64)
    stdx = np.std(Xtrain,axis=0,dtype=np.float64)
    
    Xtrain = np.divide(Xtrain-np.tile(mx,[Ntrain,1]),np.tile(stdx,[Ntrain,1]))
    Xtest = np.divide(Xtest-np.tile(mx,[Ntest,1]),np.tile(stdx,[Ntest,1]))
    
    #==============================================================================
    # # Also we could have used:
    # from sklearn import preprocessing
    # scaler = preprocessing.StandardScaler().fit(Xtrain)
    # Xtrain = scaler.transform(Xtrain)            
    # Xtest = scaler.transform(Xtest)       
    #==============================================================================

#################################################################
###################### CLASSIFIERS ##############################
#################################################################


#%% Logistic Regression

# We are going to obtain the scores for logistic regression alone,
# and also, using bagging and boosting.


if (LR_cl == 1):
    
    scores = np.empty((16))
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(Xtrain,Ytrain)


    print('Logistic Regression, train: {0:.02f}% '.format(lr.score(Xtrain,Ytrain)*100))
    print('Logistic Regression, test: {0:.02f}% '.format(lr.score(Xtest,Ytest)*100))
    
    blr = BaggingClassifier(LogisticRegression(),n_estimators = 20, max_samples=0.8,n_jobs=-1)
    blr.fit(Xtrain,Ytrain)
    scores[0] = blr.score(Xtrain,Ytrain)
    scores[1] = blr.score(Xtest,Ytest)

    # LDA
    from sklearn.lda import LDA
    lda = LDA()
    lda.fit(Xtrain,Ytrain)

    print('LDA, train: {0:.02f}% '.format(lda.score(Xtrain,Ytrain)*100))
    print('LDA, test: {0:.02f}% '.format(lda.score(Xtest,Ytest)*100))
    
    blda = BaggingClassifier(LDA(),n_estimators = 20, max_samples = 0.8,n_jobs=-1, bootstrap = False,max_features = 1.0, bootstrap_features = False)
    blda.fit(Xtrain,Ytrain)
    scores[2] = blda.score(Xtrain,Ytrain)
    scores[3] = blda.score(Xtest,Ytest)
    print('Bagging LDA, train: {0:.02f}% '.format(scores[2]*100))
    print('Bagging LDA, test: {0:.02f}% '.format(scores[3]*100))
    

    # GNB
    nb = GaussianNB()
    nb.fit(Xtrain,Ytrain)
    print('Gaussian Naive Bayes, train: {0:.02f}% '.format(nb.score(Xtrain,Ytrain)*100))
    print('Gaussian Naive Bayes, test: {0:.02f}% '.format(nb.score(Xtest,Ytest)*100))
    
    bnb = BaggingClassifier(GaussianNB(),n_estimators = 20, max_samples = 0.8,n_jobs=-1)
    bnb.fit(Xtrain,Ytrain)
    scores[4] = bnb.score(Xtrain,Ytrain)
    scores[5] = bnb.score(Xtest,Ytest)


    # Bagging kNN
    bknn = BaggingClassifier(KNeighborsClassifier(n_neighbors=4),n_jobs=-1)
    bknn.fit(Xtrain,Ytrain)
    scores[6] = bknn.score(Xtrain,Ytrain)
    scores[7] = bknn.score(Xtest,Ytest)
 

    #%% Bagging with SVM
    bsvml = BaggingClassifier(SVC(kernel = 'linear',C = 2.15),n_estimators=20,max_samples = 0.8,oob_score=True,n_jobs=-1)
    bsvmp = BaggingClassifier(SVC(kernel = 'poly',C = 215.44),n_estimators=20,max_samples = 0.8,oob_score=True,n_jobs=-1)
    bsvmr = BaggingClassifier(SVC(kernel = 'rbf',C = 215.44,  gamma = 0.002),n_estimators=20,max_samples = 0.8,oob_score=True,n_jobs=-1)
    bsvml.fit(Xtrain,Ytrain)
    bsvmp.fit(Xtrain,Ytrain)
    bsvmr.fit(Xtrain,Ytrain)
    
    scores[8] = bsvml.score(Xtrain,Ytrain)
    scores[9] = bsvml.score(Xtest,Ytest)
    scores[10] = bsvmp.score(Xtrain,Ytrain)
    scores[11] = bsvmp.score(Xtest,Ytest)
    scores[12] = bsvmr.score(Xtrain,Ytrain)
    scores[13] = bsvmr.score(Xtest,Ytest)
    
    btestscores = [bsvml.score(Xtest,Ytest),bsvmp.score(Xtest,Ytest),bsvmr.score(Xtest,Ytest)]
    maxtest = np.amax(btestscores)

    stkfold = StratifiedKFold(Ytrain, n_folds = 5)
    
    # The score function is the one we want to minimize or maximize given the label and the predicted.
    acc_scorer = make_scorer(accuracy_score)

    param_grid = dict()
    param_grid.update({'max_features':[None,'auto']})
    param_grid.update({'max_depth':np.arange(1,21)})
    param_grid.update({'min_samples_split':np.arange(2,11)})
    gtree = GridSearchCV(DecisionTreeClassifier(),param_grid,scoring=acc_scorer,cv=stkfold,refit=True,n_jobs=-1)
    gtree.fit(Xtrain,Ytrain)

    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100,max_features=gtree.best_estimator_.max_features,max_depth=gtree.best_estimator_.max_depth,min_samples_split=gtree.best_estimator_.min_samples_split,oob_score=True,n_jobs=-1)
    rf.fit(Xtrain,Ytrain)
    scores[14] = rf.score(Xtrain,Ytrain)
    scores[15] = rf.score(Xtest,Ytest)
 
    
    
alg = ['LR','LDA', 'GNB','KNN','SVM linear', 'SVM poly', 'SVM rbf', "RF"]
trainsc = [scores[0],scores[2], scores[4],scores[6],scores[8], scores[10],scores[12],scores[14] ]
testsc = [scores[1],scores[3], scores[5], scores[7],scores[9],scores[11],scores[13],scores[15]]
plt.figure()
plt.bar(np.arange(8)+0.2,trainsc,width=0.4,color='#230439',align='center')
plt.bar(np.arange(8)+0.6,testsc,width=0.4,color='#E70000',align='center')
plt.xticks(np.arange(8)+0.4,alg)
plt.title('Bagging Accuracy')
plt.ylabel('Accuracy')
plt.ylim([0,1])
plt.legend(['Train','Test'])
plt.show()

    
    kernels = ['Linear\nC=2.15','Polynomic\n C=215.44\n degree=2','Gaussian\n C=215.44\n gamma=0.002']




