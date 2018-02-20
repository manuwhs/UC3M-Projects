

import pandas as pd
import numpy as np
import scipy as sp
import utilities as util
import utilities_2 as util2
import utilities_3 as util3
import utilities_4 as util4
import utilities_5 as util5
import time
import gc
import pylab as pl
import matplotlib as plt

    
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE


load_data = 1   # Flag to load the data

if (load_data == 1):
    # Load Auctions structure
    Auctions = util2.load_pickle("Auctions",5)
    gc.collect()
    Bidders = util2.load_pickle("Bidders",5)  
    gc.collect()
    # Load ids of auctions and bidders (they were transformed into numbers)
    Auction_id = util2.load_pickle("Auction_id",1)
    Bidder_id = util2.load_pickle("Bidder_id",1)  
    
    # Load train and test bidder_ids and properties
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    
    #==============================================================================
    # # FIND TRAIN AND TEST Bidders inside the DATASET
    # # Create DICTIONARY FOR BIDDERS AND AUCTIONS to the indexes of structures
    #==============================================================================
    Bidders_dic = {};     # When you introduce the bidder_id, it outputs its index
    for i in range(len(Bidder_id)):
        Bidders_dic[Bidder_id[i]] = i
        
    Auctions_dic = {};     # When you introduce the auction name, it outputs its index
    for i in range(len(Auction_id)):
        Auctions_dic[Auction_id[i]] = i
        
    #==============================================================================
    #  Eliminate the users from the training set that are not in the dataset
    #==============================================================================
    non_existing_bidders_indx = [];
    for i in range(len(train)):
        try:
            aux = Bidders_dic[train.bidder_id[i]]
        except KeyError:
            non_existing_bidders_indx.append(i)
            
    train.drop(train.index[non_existing_bidders_indx],inplace=True)
    train.index = range(len(train))
    #==============================================================================
    #  Eliminate the users from the testing set that are not in the dataset
    #==============================================================================
    non_existing_bidders_indx = [];
    for i in range(len(test)):
        try:
            aux = Bidders_dic[test.bidder_id[i]]
        except KeyError:
            non_existing_bidders_indx.append(i)
            
    test.drop(test.index[non_existing_bidders_indx],inplace=True)
    test.index = range(len(test))

time_gaps_transform = 1

if (time_gaps_transform == 1):
# **************************************************************************
# ************************** TIME TRANSFORMATION ***************************
# **************************************************************************
    
    first_gap_center = 9.67 * np.power(10,15)  # More or less, a ojo
    second_gap_center = 9.73 * np.power(10,15)
    gaps_length, Second_column_pos, num_bids_column = util3.get_timeGapsLength(Auctions, first_gap_center, second_gap_center)
    print "Gaps Obtained"
    
    Auction_timeType = util3.get_Auctions_timeType(Auctions)
    Bidder_timeType = util3.get_Bidders_timeType(Bidders, first_gap_center, second_gap_center)
    
    print np.sum(Auction_timeType)
    print np.sum(Bidder_timeType)

    
    time1 = 50021105263158  # to save time
    time2 = 50021105263158 
    
    util3.remove_time_gaps(Auctions, Bidders,first_gap_center, second_gap_center, time1, time2)
    print "Gaps removed "

#==============================================================================
# GET THE TRAINING BIDDERS THAT HAVE MORE THAN 100 bids, the rest are humans
#==============================================================================
number_of_bids = util4.get_number_of_Bids(Bidders)
train_selection = train.copy(deep=True)
very_human_indx = [];
very_human_bot_indx = [];
only_one_bid = [];
for i in range(len(train)):
    indx = Bidders_dic[train.bidder_id[i]]
    if (number_of_bids[indx] == 1):
        only_one_bid.append(i)
    if (number_of_bids[indx] < 100):
        very_human_indx.append(i)
        if (train.outcome[i] > 0.5): # If it is a bot
            very_human_bot_indx.append(indx)
    
train_selection.drop(train.index[very_human_indx],inplace=True)
train_selection.index = range(len(train_selection))

print util4.get_bot_number(train,Bidders_dic)
print util4.get_bot_number(train_selection,Bidders_dic)

#==============================================================================
# GET THE TEST BIDDERS THAT HAVE MORE THAN 100 bids, the rest are humans
#==============================================================================

test_selection = test.copy(deep=True)
very_human_indx2 = [];
for i in range(len(test)):
    indx = Bidders_dic[test.bidder_id[i]]
    if (number_of_bids[indx] < 100):
        very_human_indx2.append(i);
        
test_selection.drop(test.index[very_human_indx2],inplace=True)

# test_selection.index = range(len(test_selection))
# Dont change index, this way we will be able to know their global position in the future.


#==============================================================================
# 
# ## Get training vectors and labels
# features_all = []
# 
# N_divs = [1, 2, 5, 10, 20, 50, 100];
# 
# own_time_distance_previous_intra_auctions = util3.get_own_time_distance_previous_intra_auctions(Bidders,Auctions)
# time_distance_previous = util3.get_time_distance_previous (Bidders,Auctions);
# own_time_distance_previous_inter_auctions = util3.get_own_time_distance_previous_inter_auctions(Bidders,Auctions)
# mybid_lastbid_distance = util3.get_mybid_lastbid_distance(Bidders,Auctions)
#    
# for N_div in N_divs:
#     
#     print N_div 
#     # Get features
#     probs_list_0,means_list_0,vars_list_0 = util4.get_meanVarAndProbs_by_ranges(own_time_distance_previous_intra_auctions,N_div,"linear")
#     
#     probs_list_1,means_list_1,vars_list_1 = util4.get_meanVarAndProbs_by_ranges(own_time_distance_previous_inter_auctions,N_div,"linear")
# 
#     probs_list_2,means_list_2,vars_list_2 = util4.get_meanVarAndProbs_by_ranges(time_distance_previous,N_div,"linear")
#     
#     probs_list_3,means_list_3,vars_list_3 = util4.get_meanVarAndProbs_by_ranges(mybid_lastbid_distance,N_div,"linear")
#     
#     # Add them to the array
#     for i in range (min(len(probs_list_0),10)):
#         features_all.append(probs_list_0[i])
#         features_all.append(means_list_0[i])
#         features_all.append(vars_list_0[i])
#         
#     for i in range (min(len(probs_list_1),10)):
#         features_all.append(probs_list_1[i])
#         features_all.append(means_list_1[i])
#         features_all.append(vars_list_1[i])
#         
#     for i in range (min(len(probs_list_2),10)):
#         features_all.append(probs_list_2[i])
#         features_all.append(means_list_2[i])
#         features_all.append(vars_list_2[i])
#         
#     for i in range (min(len(probs_list_3fac),10)):
#         features_all.append(probs_list_3[i])
#         features_all.append(means_list_3[i])
#         features_all.append(vars_list_3[i])
#         
# number_of_auctions = util4.get_number_of_Auctions(Bidders)
# proportion_consecutive_bids = util4.get_proportion_consecutive_bids(Auctions,Bidders)
# number_of_bids = util4.get_number_of_Bids(Bidders)
# 
# features_all.append(number_of_auctions)
# features_all.append(proportion_consecutive_bids)
# features_all.append(number_of_bids)
# 
# 
# # Obtain the ones just for the Selected train 
# features_all_np = np.array(features_all).T
# n_sa, n_features = features_all_np.shape
# xtrain = np.zeros((len(train_selection),n_features))
# ytrain = np.zeros(len(train_selection))
# 
# for i in range (len(train_selection)):
#     indx = Bidders_dic[train_selection.bidder_id[i]]
# # # PUTO CONTROL + 4 de los cojones que no coje estas 3 instrucciones !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#==============================================================================
#    ytrain[i] = train_selection.outcome[i]
#    xtrain[i] = features_all_np[indx][:]
#xtrain = xtrain



##################################################################
##################################################################
# FEATURE SELECTION 
##################################################################
##################################################################
#==============================================================================

#==============================================================================


#util2.store_pickle ("X", features_all_np, partitions = 1)
#util2.store_pickle ("Bidder_timeType", Bidder_timeType, partitions = 1)
#util2.store_pickle ("Bidders_dic", [Bidders_dic], partitions = 1)
#util2.store_pickle ("train_selection", [train_selection], partitions = 1)
util2.store_pickle ("test_selection", [test_selection], partitions = 1)


#==============================================================================
# # Recursive Feature Elimination
# est = SVC(kernel='linear')
# rfe = RFE(est)
# rfe.fit(selectes_xtrain,ytrain)
# rfe_scores = 1/rfe.ranking_.astype(np.float)
# rfe_index = np.argsort(rfe_scores)[::-1]
# rfe_scores = rfe_scores[rfe_index]
# rfe_scores = rfe_scores/np.max(rfe_scores)
# print('RFE Train Accuracy: '+str(rfe.score(xtrain,ytrain)))
# print('RFE Test Accuracy: '+str(rfe.score(xtest,ytest)))
# 
# pl.figure()
# pl.scatter(rfe_index,rfe_scores,color="blue")
#==============================================================================





#util4.plot_2_features_train(train_selection, Bidders_dic, probs_list_0[0], probs_list_1[3])

#prediction = util5.system_random (number_of_bids, Bidders_dic)
#util5.write_csv_output(prediction, "pene.csv")