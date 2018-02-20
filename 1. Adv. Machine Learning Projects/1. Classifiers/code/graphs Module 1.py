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
#==============================================================================
#     plt.figure()
#     
#     ax = plt.subplot(1,4,1)   # First band for every pixel
#     img = data[:,1].reshape(145,145)
#     plt.imshow(img)
#     ax.set_title("Band 1",fontsize = 20)
#     
#     ax = plt.subplot(1,4,2)   # First band for every pixel
#     img = data[:,50].reshape(145,145)
#     plt.imshow(img)
#     ax.set_title("Band 50",fontsize = 20)
# 
#     ax = plt.subplot(1,4,3)   # First band for every pixel
#     img = data[:,100].reshape(145,145)
#     plt.imshow(img)
#     ax.set_title("Band 100",fontsize = 20)
# 
#     ax = plt.subplot(1,4,4)   # First band for every pixel
#     lab = labels.reshape(145,145)
#     plt.imshow(lab)
#     ax.set_title("Ground Truth",fontsize = 20)
#==============================================================================

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
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(Xtrain,Ytrain)
    
    scores = np.empty((6))
    scores[0] = lr.score(Xtrain,Ytrain)
    scores[1] = lr.score(Xtest,Ytest)
    
    print('Logistic Regression, train: {0:.02f}% '.format(scores[0]*100))
    print('Logistic Regression, test: {0:.02f}% '.format(scores[1]*100))
    

    from sklearn.lda import LDA
    lda = LDA()
    lda.fit(Xtrain,Ytrain)
    scores[2] = lda.score(Xtrain,Ytrain)
    scores[3] = lda.score(Xtest,Ytest)
    print('LDA, train: {0:.02f}% '.format(scores[2]*100))
    print('LDA, test: {0:.02f}% '.format(scores[3]*100))
    

    nb = GaussianNB()
    nb.fit(Xtrain,Ytrain)
    scores[4] = nb.score(Xtrain,Ytrain)
    scores[5] = nb.score(Xtest,Ytest)
    print('Gaussian Naive Bayes, train: {0:.02f}% '.format(scores[4]*100))
    print('Gaussian Naive Bayes, test: {0:.02f}% '.format(scores[5]*100))
    
 
    alg = ['LR','LDA', 'GNB']
    trainsc = [scores[0],scores[2], scores[4]]
    testsc = [scores[1],scores[3], scores[5]]
    plt.figure()
    plt.bar(np.arange(3)+0.2,trainsc,width=0.4,color='#230439',align='center')
    plt.bar(np.arange(3)+0.6,testsc,width=0.4,color='#E70000',align='center')
    plt.xticks(np.arange(3)+0.4,alg)
    plt.title('Simple Classifiers\' Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim([0,1])
    plt.legend(['Train','Test'])
    plt.show()


if (KNN_cl == 1):
#   KNN Classifier  Perform cross-validation
    nfold = 5
    n_K = 10
    
#    rang_k = np.zeros((nfold,n_K))
#    accval = np.zeros((nfold,n_K))
#    acctrain = np.zeros((nfold,n_K))
#    acctest = np.zeros((nfold,n_K))
#    
#    folds = StratifiedKFold(Ytrain,n_folds = nfold)
#    
#    for k in range(0,n_K):   # For each K values
#        
#        for f, (train_index, val_index) in enumerate(folds): # For every crosvalidation
#            X_train, X_val = Xtrain[train_index], Xtrain[val_index]
#            y_train, y_val = Ytrain[train_index], Ytrain[val_index]
#            print f
#            knn = KNeighborsClassifier(n_neighbors = k+1)
#            knn.fit(X_train,y_train)
#            accval[f,k] = knn.score(X_val,y_val)
#            acctrain[f,k] = knn.score(X_train,y_train)
#            acctest[f,k] = knn.score(Xtest,Ytest)
#        
#    # Get average of these fuckers
#    accval = np.mean(accval,axis = 0)
#    acctrain =  np.mean(acctrain,axis = 0)
#    acctest =  np.mean(acctest,axis = 0)
            
    bestk_idx = np.argmax(accval)
    print('Best n_neigh = '+str(rang_k[bestk_idx]))
    
    knn = KNeighborsClassifier(n_neighbors = 4)
    knn.fit(Xtrain,Ytrain)
    print('kNN, train: {0:.02f}% '.format(knn.score(Xtrain,Ytrain)*100))
    print('kNN, test: {0:.02f}% '.format(knn.score(Xtest,Ytest)*100))

    # Plot evolution of parameters performance
    plt.figure()
    plt.plot(range(1,n_K+1), acctrain,c='b',lw = 3)
    plt.plot(range(1,n_K+1), accval,c='g',lw = 3)
    plt.plot(range(1,n_K+1), acctest,c='r',lw = 3)
    
    plt.title('Accuracy KNN')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.ylim([0.6,1])
    plt.grid()
    plt.legend(['Train','Validation','Test'])
    plt.show()


    # Perform authomatic grid search
#    param_knn = dict()
#    param_knn.update({'n_neighbors':np.arange(1,11)})
#    
#    stkfold = StratifiedKFold(Ytrain, n_folds = 5)
#    # The score function is the one we want to minimize or maximize given the label and the predicted.
#    acc_scorer = make_scorer(accuracy_score)
#    
#    
#    gknn = GridSearchCV(KNeighborsClassifier(),param_knn ,scoring = acc_scorer,cv = stkfold,refit = True, n_jobs=-1)
#    gknn.fit(Xtrain,Ytrain)
#    scores = np.empty((4))
#    scores[0] = gknn.score(Xtrain,Ytrain)
#    scores[1] = gknn.score(Xtest,Ytest)
#    print('{0}-NN, train: {1:.02f}% '.format(gknn.best_estimator_.n_neighbors,scores[0]*100))
#    print('{0}-NN, test: {1:.02f}% '.format(gknn.best_estimator_.n_neighbors,scores[1]*100))
#    
#    # Plot evolution of parameters performance
#    scores = [np.mean(val[1]) for val in gknn.grid_scores_]
#    plt.figure()
#    plt.plot(np.arange(1,np.alen(scores)+1),scores,c='c',lw=2,aa=True)
#    plt.plot(np.argmax(scores)+1,np.amax(scores),'v')
#    plt.title('Accuracy KNN')
#    plt.xlabel('K')
#    plt.ylabel('Accuracy')
#    plt.grid()
#    plt.show()

    
    

#%% SVM Classifier
# Params C, kernel, degree, params of kernel

if (SVM_cl == 1):
    
    # Parameters for the validation
    C = np.logspace(-3,5,30)
    p = np.arange(1,5)
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
    stkfold = StratifiedKFold(Ytrain, n_folds = 4)
    
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

#    gsvml = GridSearchCV(SVC(class_weight='auto'),param_grid_linear, scoring = acc_scorer,cv = stkfold, refit = True,n_jobs=-1)
    gsvmp = GridSearchCV(SVC(class_weight='auto'),param_grid_pol, scoring = acc_scorer,cv = stkfold, refit = True,n_jobs=-1)
#    gsvmr = GridSearchCV(SVC(class_weight='auto'),param_grid_rbf, scoring =acc_scorer,cv = stkfold, refit = True,n_jobs=-1)
#    
#    gsvml.fit(Xtrain,Ytrain)
    gsvmp.fit(Xtrain,Ytrain)
#    gsvmr.fit(Xtrain,Ytrain)
    
#    trainscores = [gsvml.score(Xtrain,Ytrain),gsvmp.score(Xtrain,Ytrain),gsvmr.score(Xtrain,Ytrain)]
#    testscores = [gsvml.score(Xtest,Ytest),gsvmp.score(Xtest,Ytest),gsvmr.score(Xtest,Ytest)]
#    maxtrain = np.amax(trainscores)
#    maxtest = np.amax(testscores)
#    print('SVM, train: {0:.02f}% '.format(maxtrain*100))
#    print('SVM, test: {0:.02f}% '.format(maxtest*100))
    
#    #%% Bagging with SVM
#    bsvml = BaggingClassifier(gsvml.best_estimator_,n_estimators=100,oob_score=True,n_jobs=-1)
#    bsvmp = BaggingClassifier(gsvmp.best_estimator_,n_estimators=100,oob_score=True,n_jobs=-1)
#    bsvmr = BaggingClassifier(gsvmr.best_estimator_,n_estimators=100,oob_score=True,n_jobs=-1)
#    
#    bsvml.fit(Xtrain,Ytrain)
#    bsvmp.fit(Xtrain,Ytrain)
#    bsvmr.fit(Xtrain,Ytrain)
#    btrainscores = [bsvml.score(Xtrain,Ytrain),bsvmp.score(Xtrain,Ytrain),bsvmr.score(Xtrain,Ytrain)]
#    maxtrain = np.amax(btrainscores)
#    btestscores = [bsvml.score(Xtest,Ytest),bsvmp.score(Xtest,Ytest),bsvmr.score(Xtest,Ytest)]
#    maxtest = np.amax(btestscores)
#    print('Bagged SVM, train: {0:.02f}% '.format(maxtrain*100))
#    print('Bagged SVM, test: {0:.02f}% '.format(maxtest*100))
    
#    kernels = ['Linear\nC=2.15','Polynomic\n C=215.44\n degree=2','Gaussian\n C=215.44\n gamma=0.002']
#    plt.figure()
#    plt.subplot(1,2,1)
#    plt.bar(np.arange(3)+0.2,trainscores,width=0.2,color='c',align='center')
#    plt.bar(np.arange(3)+0.4,testscores,width=0.2,color='r',align='center')
#    plt.xticks(np.arange(3)+0.3,kernels)
#    plt.title('SVM Accuracy')
#    plt.ylabel('Accuracy')
#    plt.legend(['Train','Test'])
#    plt.subplot(1,2,2)
#    plt.bar(np.arange(3)+0.2,btrainscores,width=0.2,color='c',align='center')
#    plt.bar(np.arange(3)+0.4,btestscores,width=0.2,color='r',align='center')
#    plt.xticks(np.arange(3)+0.3,kernels)
#    plt.title('Bagged SVM Accuracy')
#    plt.ylabel('Accuracy')
#    plt.legend(['Train','Test'])
#    plt.show()


    # Plot evolution of the C parameter performance
    # Plot evolution of parameters performance

#    validate = gsvml.grid_scores_
    validate = gsvmp.grid_scores_
    
#    validation_error = np.zeros((len(validate),1))
#    
#    for i in range (len(validate)):
#        validation_error[i] = validate[i][1]

    validation_error = np.zeros((4,30))
    
    for i in range (30):
        for j in range (4):
            validation_error[j][i] = validate[i*4 + j][1]
        
    plt.figure()
    for i in range(4):
        plt.plot(C, validation_error[i],lw = 3)
        
    plt.title('Accuracy Polynomic SVM')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.ylim([0,1])
    plt.xscale('log')
    plt.grid()
    plt.legend(['d = 1','d = 2','d = 3','d = 4'])
    plt.show()


#%% Tree Classifier

if (Tree_cl == 1):
    
    stkfold = StratifiedKFold(Ytrain, n_folds = 5)
    
    # The score function is the one we want to minimize or maximize given the label and the predicted.
    acc_scorer = make_scorer(accuracy_score)

    param_grid = dict()
    param_grid.update({'max_features':[None,'auto']})
    param_grid.update({'max_depth':np.arange(1,21)})
    param_grid.update({'min_samples_split':np.arange(2,11)})
    gtree = GridSearchCV(DecisionTreeClassifier(),param_grid,scoring=acc_scorer,cv=stkfold,refit=True,n_jobs=-1)
    gtree.fit(Xtrain,Ytrain)
    scores = np.empty((6))
    scores[0] = gtree.score(Xtrain,Ytrain)
    scores[1] = gtree.score(Xtest,Ytest)
    print('Decision Tree, train: {0:.02f}% '.format(scores[0]*100))
    print('Decision Tree, test: {0:.02f}% '.format(scores[1]*100))
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100,max_features=gtree.best_estimator_.max_features,max_depth=gtree.best_estimator_.max_depth,min_samples_split=gtree.best_estimator_.min_samples_split,oob_score=True,n_jobs=-1)
    rf.fit(Xtrain,Ytrain)
    scores[2] = rf.score(Xtrain,Ytrain)
    scores[3] = rf.score(Xtest,Ytest)
    print('Random Forest, train: {0:.02f}% '.format(scores[2]*100))
    print('Random Forest, test: {0:.02f}% '.format(scores[3]*100))
    
    # Extremely Randomized Trees
    ert = ExtraTreesClassifier(n_estimators=100,max_features=gtree.best_estimator_.max_features,max_depth=gtree.best_estimator_.max_depth,min_samples_split=gtree.best_estimator_.min_samples_split,n_jobs=-1)
    ert.fit(Xtrain,Ytrain)
    scores[4] = ert.score(Xtrain,Ytrain)
    scores[5] = ert.score(Xtest,Ytest)
    print('Extremely Randomized Trees, train: {0:.02f}% '.format(scores[4]*100))
    print('Extremely Randomized Trees, test: {0:.02f}% '.format(scores[5]*100))
    
    # Plot results
    alg = ['Decision Tree','Random Forest','Extremely Randomized Trees']
    plt.figure()
    trainsc = [scores[0],scores[2],scores[4]]
    testsc = [scores[1],scores[3],scores[5]]
    plt.bar(np.arange(3)+0.2,trainsc,color='#230439',width=0.2,align='center')
    plt.bar(np.arange(3)+0.4,testsc,color='#E70000',width=0.2,align='center')
    plt.xticks(np.arange(3)+0.3,alg)
    plt.title('Trees accuracy')
    plt.ylabel('Accuracy')
    plt.legend(['Train','Test'])
    plt.show()
    
    #dot_data = StringIO()
    #tree.export_graphviz(gs.best_estimator_,out_file=dot_data)
    #graph = pd.graph_from_dot_data(dot_data.getvalue())
    #graph.write_pdf("tree.pdf")

#%% Boosted Trees

if (BT_cl == 1):
    ab = AdaBoostClassifier(n_estimators=100)
    ab.fit(Xtrain,Ytrain)
    scores = np.empty((4))
    scores[0] = ab.score(Xtrain,Ytrain)
    scores[1] = ab.score(Xtest,Ytest)
    print('Boosted Stump, train: {0:.02f}% '.format(scores[0]*100))
    print('Boosted Stump, test: {0:.02f}% '.format(scores[1]*100))
    
    abtree = AdaBoostClassifier(gtree.best_estimator_,n_estimators=500)
    abtree.fit(Xtrain,Ytrain)
    scores[2] = abtree.score(Xtrain,Ytrain)
    scores[3] = abtree.score(Xtest,Ytest)
    print('Boosted Trees, train: {0:.02f}% '.format(scores[2]*100))
    print('Boosted Trees, test: {0:.02f}% '.format(scores[3]*100))
    
    alg = ['Boosted Stump','Boosted Trees']
    trainsc = [scores[0],scores[2]]
    testsc = [scores[1],scores[3]]
    plt.figure()
    plt.bar(np.arange(2)+0.2,trainsc,width=0.4,color='c',align='center')
    plt.bar(np.arange(2)+0.6,testsc,width=0.4,color='r',align='center')
    plt.xticks(np.arange(2)+0.4,alg)
    plt.title('Boosting accuracy')
    plt.ylabel('Accuracy')
    plt.legend(['Train','Test'])
    plt.show()
    
    real_test_errors = []
    real_train_errors = []
    for real_train_predict,real_test_predict in zip(abtree.staged_predict(Xtrain),abtree.staged_predict(Xtest)):
        real_train_errors.append(1-accuracy_score(real_train_predict, Ytrain))
        real_test_errors.append(1-accuracy_score(real_test_predict, Ytest))
        
    plt.figure()
    #plt.plot(range(1, len(real_train_errors) + 1),real_train_errors,c='c',lw=2)
    plt.plot(range(1, len(real_test_errors) + 1),real_test_errors,c='r',lw=2)
    plt.ylabel('Test Error')
    plt.xlabel('Number of Trees')
    plt.title('Staged test errors for Boosted Trees')
    plt.legend(['Train','Test'])
    plt.grid()
    plt.show()
