import numpy as np
import copy

from sklearn.cross_validation import StratifiedShuffleSplit

class Bagging_manu():
    def __init__(self, est, max_samples = 0.8, n_estimators = 10, ensemble_type = "average"):
        self.est = est
        self.max_samples = max_samples
        self.n_estimators = n_estimators
        self.ensemble_type = ensemble_type
        self.baggers = []; # Array of the bagger object
        
# Split the dataset into equeall
    def fit(self, Xtrain, Ytrain):
        # We create different weighted splits of the training set.
        # Every split will be used to train a bagger
        SSS =  StratifiedShuffleSplit(Ytrain, self.n_estimators, self.max_samples, random_state = 0)
        for Selected_index, NonSelected_index in SSS:   #  For e
            bagger = copy.deepcopy(self.est)
            X_sel = Xtrain[Selected_index]
            Y_sel = Ytrain[Selected_index]
            bagger.fit(X_sel,Y_sel)
            self.baggers.append(bagger)
    
    def predict_proba(self,X):
        bagging_proba = 0;
        for bagger in self.baggers:
#            print bagging_proba
            bagging_proba += bagger.predict_proba(X)  # (n_samples x n_classes)
            
        bagging_proba/= len(self.baggers)     # Average proba of the classfiers
        return bagging_proba
            
    """
    The predicted class of an input sample is computed as the class with the highest mean predicted probability. 
    If base estimators do not implement a predict_proba method, then it resorts to voting.
    """
    def predict(self,X):
        bagging_proba = self.predict_proba(X)
        predicted = np.argmax(bagging_proba, axis = 1 ) # Obtain for every sample, the class with the highest probability
#        print predicted
        return predicted;
        
    def score(self,X,Y):
        predicted = self.predict(X)
        N_samples = len(Y)
        score = 0.0;
        for i in range (N_samples):
            print predicted[i], Y[i]
            if (Y[i] == predicted[i]):
                score += 1;
        
        score /= N_samples;
        return score;
    

    




