# -*- coding: utf-8 -*-
"""
Created on Mon Mar 02 19:24:42 2015

@author: Dani
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.svm import LinearSVC,SVC
from sklearn.lda import LDA

from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.naive_bayes import GaussianNB

def classify(Xtrain,Xtest,Ytrain,Ytest):
    '''
    Linear and RBF SVM classifiers
    '''
    scores = np.zeros((5,))
    

    lr = LogisticRegression()
    lr.fit(Xtrain,Ytrain)
    scores[0] = lr.score(Xtest,Ytest)

    lda = LDA()
    lda.fit(Xtrain,Ytrain)
    scores[1] = lda.score(Xtest,Ytest)

    nb = GaussianNB()
    nb.fit(Xtrain,Ytrain)
    scores[2] = nb.score(Xtest,Ytest)
    
    lsvm = LinearSVC( C = 1)
    lsvm.fit(Xtrain,Ytrain)
    scores[3] = lsvm.score(Xtest,Ytest)
    
    gsvm = SVC(kernel='rbf', C = 1000)
    gsvm.fit(Xtrain,Ytrain)
    scores[4] = gsvm.score(Xtest,Ytest)
    return scores

def colorRandomizer():
    rgbColors = np.random.rand(1,3)
    return rgbColors;

def plotAccuracy(fig,nComponents,scores):
    plt.plot(nComponents,scores[0,:],figure=fig,c='c',lw=2,label='Linear SVM')
    plt.plot(nComponents,scores[1,:],figure=fig,c='r',lw=2,label='RBF SVM')
    plt.xlim(1,np.amax(nComponents))
    plt.xlabel('number of components')
    plt.ylabel('accuracy')
    plt.legend(loc='upper left')
    plt.grid(True)
    
def plotData(fig,X,Y,classColors):
    ndims = np.shape(X)[1]
    labels = np.unique(Y.astype(np.int))
    if ndims==2:
        for i,l in enumerate(labels):
            plt.scatter(X[Y==l,0],X[Y==l,1],alpha=0.5,figure=fig,c=classColors[i,:])
    elif ndims==3:
        ax = fig.add_subplot(111, projection='3d')
        for i,l in enumerate(labels):
            ax.scatter(X[Y==l,0],X[Y==l,1],X[Y==l,2],c=classColors[i,:])
    
    