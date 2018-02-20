

import pandas as pd
import numpy as np
import scipy as sp
import utilities as util
import utilities_2 as util2
import utilities_3 as util3
import utilities_4 as util4
import time
import gc
import pylab as pl
    

#==============================================================================
# # CREATE NUMPY ARRAYS FOR EACH AUCTIONS FOR THE TIMES AND PREPROCESS THEM
#==============================================================================

# We can see that:
# Some users bet on the first and second column only
# Others only bets in the third column.
# Same for aucitons 

# We can totally separate columns 1,2 from 3. !!!

# Could it be possible that column 2 times have been divided by 2 ?

#==============================================================================
# Times_a = []
# for i in range(len(Auctions)):
#     time_a = np.array(Auctions[i][2],dtype = float)
#     time_a = (time_a)/(np.power(10,9)) 
#     # We use the last bid as reference, coz usually everyone bids at the start
#     # but not many bid at the beggining
#     Times_a.append(time_a)
#==============================================================================


#print util4.get_bot_number(train,Bidders_dic)
#print util4.get_bot_number(train_selection,Bidders_dic)

#==============================================================================
# # STUDY OF HOW THE TIMES BEHAVES!! 
#==============================================================================
# For an auction, we can view different bids from different users (bot and non-bot)
#util3.plot_auction_times (100, Auctions,train, Times_a, Bidder_id)

#==============================================================================
#  Get all the biddinds of an user, no matter the auction
#==============================================================================
#own_times_bidding = util3.get_own_times_bidding(Bidders,Auctions)
#util3.plot_human_bot_train_times_data(train_selection,Bidders_dic,own_times_bidding)

#==============================================================================
# Get the Time distance between an User's bid and the previous bid
#     - For every user we create an array with these distances
#     - Obtained from Auctions
#     - First we have to create an empty array of n_bidders lists
#==============================================================================

#time_distance_previous = util3.get_time_distance_previous (Bidders,Auctions, Times_a);
#util3.plot_human_bot_train_times_data(train_selection,Bidders_dic,time_distance_previous)

#==============================================================================
# Get the Time distance between an User's bid and uts own last bid made:
#     - WE COULD DO IT AUCTION BY AUCTION OR IN TOTAL. WE CHOOSE 2:
#            Intra-Auctions average.
#            Global
#     - For every user we create an array with these distances
#     - Obtained from Bidders
#     - For the global, we have to join all the auction bids in a single array and reorder it
#     - First we have to create an empty array of n_bidders lists
#==============================================================================
#own_time_distance_previous_intra_auctions = util3.get_own_time_distance_previous_intra_auctions(Bidders,Auctions)
#util3.plot_human_bot_train_times_data(train_selection,Bidders_dic,own_time_distance_previous_intra_auctions)

#own_time_distance_previous_inter_auctions = util3.get_own_time_distance_previous_inter_auctions(Bidders,Auctions)
#util3.plot_human_bot_train_times_data(train,Bidders_dic,own_time_distance_previous_inter_auctions)


#==============================================================================
#  Get the time distance from the last bid to my last bid
#   - To do that, for every Bidder, we look to its last bid for every object
#   - Get the object_indx and search in Auctions for the las bid.
#   - Just delete values
#==============================================================================
#mybid_lastbid_distance = util3.get_mybid_lastbid_distance(Bidders,Auctions)
#util3.plot_human_bot_train_times_data(train,Bidders_dic,mybid_lastbid_distance)

# Features !!!
#==============================================================================
# number_of_bids = util4.get_number_of_Bids(Bidders)
# number_of_auctions = util4.get_number_of_Auctions(Bidders)
# number_of_auctions_won = util4.get_number_of_Auctions_won(Auctions,Bidders)
#==============================================================================

#==============================================================================
# GET THE TRAINING BIDDERS THAT HAVE MORE THAN 100 bids, the rest are humans
#==============================================================================
#train_selection = train.copy(deep=True)
#
#very_human_indx = [];
#very_human_bot_indx = [];
#only_one_bid = [];
#for i in range(len(train)):
#    indx = Bidders_dic[train.bidder_id[i]]
#    if (number_of_bids[indx] == 1):
#        only_one_bid.append(i)
#    if (number_of_bids[indx] < 100):
#        very_human_indx.append(i)
#        if (train.outcome[i] > 0.5): # If it is a bot
#            very_human_bot_indx.append(indx)
#    
#train_selection.drop(train.index[very_human_indx],inplace=True)
#train_selection.index = range(len(train_selection))

# Try to see properties of the bot with very few bids.
#==============================================================================
# for i in range (len(very_human_bot_indx)):   # Get their countries
#     print Bidders[very_human_bot_indx[i]][3]
#     
# for i in range (len(very_human_bot_indx)):   # Get IPs
#     print Bidders[very_human_bot_indx[i]][2]
#     
# for i in range (len(very_human_bot_indx)):  # Get their type of good
#     print Auctions[Bidders[very_human_bot_indx[i]][1][0][0]][1]
#     
#util3.plot_Auctions_bots_less100 (only_one_bid[:100], Auctions, Bidders)
#==============================================================================

print "End"

#==============================================================================
# test_selection = test.copy(deep=True)
# 
# very_human_indx = [];
# for i in range(len(test)):
#     indx = Bidders_dic[test.bidder_id[i]]
#     if (number_of_bids[indx] < 100):
#         very_human_indx.append(i)
#     
# test_selection.drop(test.index[very_human_indx],inplace=True)
# test_selection.index = range(len(test_selection))
#==============================================================================
    
#==============================================================================
# # BIDS DISTRIBUTIONS !!!!
#==============================================================================
#==============================================================================
#  Distribution all bids !
#probs = util3.get_allBids_Distribution(Auctions, 2000)
#probs = probs / np.sum(probs)
#time_inst = np.zeros(len(probs));
#min_t = 9.631 * np.power(10,15)
#max_t = 9.672 * np.power(10,15)
#for i in range (len(probs)):
#    time_inst[i] = min_t + (max_t - min_t)*(i)/len(probs)
#pl.plot(time_inst,probs,color="black",linewidth=2.0)
 

#get_AuctionsEnd = util4.get_AuctionsEnd(Auctions)
#get_AuctionsStart = util4.get_AuctionsStart(Auctions)
#
#probs = util3.get_Bids_Distribution(get_AuctionsStart, 200)
#time_inst = np.zeros(len(probs));
#min_t = 9.631 * np.power(10,15)
#max_t = 9.672 * np.power(10,15)
#for i in range (len(probs)):
#    time_inst[i] = min_t + (max_t - min_t)*(i)/len(probs)
#
#pl.plot(range(len(probs)),probs,color= '#66cc00', linewidth=3.0)
#probs = util3.get_Bids_Distribution(get_AuctionsEnd, 200)
#pl.plot(range(len(probs)),probs,color="black",linewidth=3.0)
#pl.grid(True)

    
    
    
# # Bids distribution of an auction
# probs = util3.get_Bids_Distribution(Auctions[3][2], 10)
# pl.plot(range(len(probs)),probs,color="blue")
#==============================================================================

# Distribution of the start and ending of auctions
#AuctionsLength = util4.get_AuctionsLength(Auctions)
#get_AuctionsEnd = util4.get_AuctionsEnd(Auctions)
#get_AuctionsStart = util4.get_AuctionsStart(Auctions)
#
#pl.figure("Start and Ending of Auction Distrib")
#probs = util3.get_Bids_Distribution(get_AuctionsStart, 200)
#pl.plot(range(len(probs)),probs,color="blue", linewidth=3.0)
#probs = util3.get_Bids_Distribution(get_AuctionsEnd, 200)
#pl.plot(range(len(probs)),probs,color="red",linewidth=3.0)
#pl.grid(True)
#
#pl.figure("Duration Distrib")
#probs = util3.get_Bids_Distribution(AuctionsLength, 200)
#pl.plot(range(len(probs)),probs,color="red")

# Distribution of the bids of the Selected Users
#own_times_bidding = util3.get_own_times_bidding(Bidders,Auctions);
#n_div = 200
#for sel_bidder_indx in range (1,600):
#    indx = Bidders_dic[train_selection.bidder_id[sel_bidder_indx]]
#    bidds = own_times_bidding[indx]
#    probs = util3.get_Bids_Distribution(bidds, n_div)
#    
#    time_instances =  bidds[0] + ((bidds[-1] - bidds[0])/n_div) * np.array(range(n_div))
#    if (train_selection.outcome[sel_bidder_indx] < 0.5):
#        pl.figure("human")
#        pl.plot(time_instances,probs,color="blue")
#    else:
#        pl.figure("bot")
#        pl.plot(time_instances,probs,color="red")

number_of_auctions = util4.get_number_of_Auctions(Bidders)
number_of_bids = util4.get_number_of_Bids(Bidders)

#util4.plot_2_features_train(train, Bidders_dic, number_of_bids, number_of_auctions_won)
#util4.plot_2_features_train(train, Bidders_dic, number_of_auctions, number_of_auctions_won)
util4.plot_2_features_train(train, Bidders_dic, number_of_bids, number_of_auctions)
pl.grid(True)




