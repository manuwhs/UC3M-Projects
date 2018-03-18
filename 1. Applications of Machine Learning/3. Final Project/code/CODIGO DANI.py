
import pickle
import numpy as np
import matplotlib as plt
import pylab as pl

## ELIMINAR TIEMPOS !!!!
    
    first_gap_center = 9.67 * np.power(10,15)  # More or less, a ojo
    second_gap_center = 9.73 * np.power(10,15)
#    time1,time2,nums, Data_time_length = get_timeGapsLength(Auctions)
    time1 = 50021105263158  # to save time
    time2 = 50021105263158 
    
    ### DANI !!!!!!!
    ### DANI !!!!!!!
    ## bids[j] es el tiempo de cualquier bid !
    for j in range(len(bids)):
        if (bids[j] > first_gap_center):
                if (bids[j] > second_gap_center):
                    bids[j] -= time2
                bids[j] -= time1

## OBTENER USUARIOS > 100 y Limpiar los que no están.


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
    
    
#==============================================================================
# GET THE TRAINING BIDDERS THAT HAVE MORE THAN 100 bids, the rest are humans
#==============================================================================
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
    