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




def classify(dataTrain,dataTest,labelsTrain,labelsTest):
    '''
    Linear and RBF SVM classifiers
    '''
    scores = np.zeros((2,))
    lsvm = LinearSVC()
    lsvm.fit(dataTrain,labelsTrain)
    scores[0] = lsvm.score(dataTest,labelsTest)
    gsvm = SVC(kernel='rbf')
    gsvm.fit(dataTrain,labelsTrain)
    scores[1] = gsvm.score(dataTest,labelsTest)
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
    
def get_bagger_data(xTrain,yTrain, prop_bots = 0.8 , ratio = 1 ):   # , n_bags = 10, prop_bots = 0.8 )
    
    # Separate the different classes
    bots = xTrain[np.where(yTrain > 0.5 )]
    n_bots, n_features = bots.shape
    
    humans = xTrain[np.where(yTrain < 0.5)]
    n_humans, n_features = bots.shape
    
    n_bots_final = prop_bots * n_bots;
    n_human_final = n_bots_final * ratio;
    
    # Create the different sets for bagging
    selected_bots = np.copy(bots)
    selected_humans = np.copy(humans)
    
#    print selected_bots.shape
#    print selected_humans.shape
    
    np.random.shuffle(selected_bots)
    np.random.shuffle(selected_humans)
    
    selected_bots = selected_bots[:n_bots_final]
    selected_humans = selected_humans[:n_human_final]
    
    bag_traininig = np.concatenate((selected_bots, selected_humans), axis = 0)
    ylabels = np.concatenate((np.zeros(len(selected_bots)), np.ones(len(selected_humans))), axis = 0)
    
    n_train = len(ylabels)
    permut = np.random.permutation(n_train)
    
    bag_traininig = bag_traininig[permut]
    ylabels = ylabels[permut]
    
    return (bag_traininig, ylabels)