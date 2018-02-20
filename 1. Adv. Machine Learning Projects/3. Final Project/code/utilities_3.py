
import pickle
import numpy as np
import matplotlib as plt
import pylab as pl

# Utilities for obtaining and plotting TIME instances vectors of bids !! 



def get_difference_array(times_a):
# This functions outputs the difference between a value the given array and the 
# previous. If the length is N, the vector returned has length N-1
    if (len(times_a) < 1):
        return np.array([])
    # To perfom sustraction, we make a copy of the times
    # If the original is [4 6 7 11]
    # or = [ 4 6 7 11 0]
    # cp = [ 0 4 6 7 11]
    # sub = [4 2 1 4 11]
    # The ones in the middle are the good ones
    time_or = np.concatenate((times_a, np.zeros((1))),axis = 0)   # Array with the times of the auction
    time_cp = np.concatenate((np.zeros((1)), times_a),axis = 0)   # Array with the times of the auction
    time_sub = (time_or - time_cp)
            
    time_sub = np.delete(time_sub, 0, 0)  # Delete first and last values
    time_sub = np.delete(time_sub, len(time_sub)-1, 0)  # Delete first and last values
    
    return time_sub

def plot_human_bot_train_times_data (train, Bidders_dic, feature_arrays):
# This function plots, for every training bidder, all the time instances in
# their time feature vector index. It created two plots:
#    - Human: For all the human bidders
#    - Bots: For all the bots biddersa
# In each graph, every bidder is differentiated in the y axis and the times in the
# x axis.
# This function can be used to get an idea of the distribution.
    
    n_bot = 0
    n_hum = 0
    for i in range (len(train)):
        print i
        indx = Bidders_dic[train.bidder_id[i]]
#        print i
        
        humans_plot = pl.figure("human")
        bots_plot = pl.figure("bots")
        
        if (train.outcome[i] < 0.5):  # Outcome of the i-th bidder in train
            pl.figure("human")
            pl.scatter(feature_arrays[indx],0.1*n_hum*np.ones((1,len(feature_arrays[indx]))),color="blue")
            n_hum += 1
        else:
            pl.figure("bots")
            pl.scatter(feature_arrays[indx],0.1*n_bot*np.ones((1,len(feature_arrays[indx]))),color="red")
            n_bot += 1
            
def plot_test_times_data (test, Bidders_dic, feature_arrays):
# Same as function plot_human_bot_train_times_data() but for the test bidders.
#There is only 1 graph since we dont know if they are bots or humans
    for i in range (len(test)):
        indx = Bidders_dic[test.bidder_id[i]]
#        print i
        unknowkn_plot = pl.figure("unknown")
        pl.scatter(feature_arrays[indx],0.1*i*np.ones((1,len(feature_arrays[indx]))),color="blue")

def plot_auction_times (num, Auctions,train, Times_a, Bidder_id):
#       Plots for every auction:
#        - The time instances where someone has made a bid. Differentiating between bot,
#        human and unknown in 3 different graphs
#       The y axis are the different auctions and the x axis are the time instances
#       given by "Times_a"

    for a in range(num):
        Bidders_ids = [ Bidder_id[i] for i in Auctions[a][3]] # Get the id of the bidders in the auction
        times_plot = Times_a[a]  # Auction we are gonna plot
        # Now we sepparate the users that made the bids into human-blue, bots-red, unknown-black
        human_times = []
        bot_times = []
        unknown_times = []
        
        for b in range(len(times_plot)):  # For evert bid in the auction
            indx = train.loc[train["bidder_id"] == Bidders_ids[b]].index.tolist()  # Get position of the user inside the training array
            if (indx != []):
                if (train.outcome[indx[0]] < 0.5):   # If it is 0, it is a human
                    human_times.append(times_plot[b])
                else:
                    bot_times.append(times_plot[b])
            else:
                unknown_times.append(times_plot[b])
        
        human_times = np.array(human_times)
        bot_times =  np.array(bot_times)
        unknown_times = np.array(unknown_times)
        
        pl.figure("human")
        pl.scatter(human_times,0.1*a*np.ones((1,len(human_times))),color="blue") # Dibujamos un scatterplot en la ventana 'scatter'
        pl.figure("bots")
        pl.scatter(bot_times,0.1*a*np.ones((1,len(bot_times))),color="red") # Dibujamos un scatterplot en la ventana 'scatter'
        pl.figure("unknown")
        pl.scatter(unknown_times,0.1*a*np.ones((1,len(unknown_times))),color="black") # Dibujamos un scatterplot en la ventana 'scatter'
#

def get_time_distance_previous (Bidders,Auctions):
    time_distance_previous = []
    for i in range(len(Bidders)):
        time_distance_previous.append([])
    
    for i in range(len(Auctions)):
        times_a = np.array(Auctions[i][2])
        
        time_or = np.concatenate((times_a, np.zeros((1))),axis = 0)   # Array with the times of the auction
        time_cp = np.concatenate((np.zeros((1)), times_a),axis = 0)   # Array with the times of the auction
        time_sub = (time_or - time_cp)

        for j in range(1,len(times_a)): # starts in 1 coz the first has no difference
            time_distance_previous[Auctions[i][3][j]].append(time_sub[j])
          # If time was inverted, then what we do is, instead of assigning the time distance between 1 and 2 2>1  to 1
          # we assing that time distance to 1 !! in the code, just type j-1
    return time_distance_previous


def get_own_time_distance_previous_intra_auctions(Bidders,Auctions):
    own_time_distance_previous_intra_auctions = []
    for i in range(len(Bidders)):
        own_time_distance_previous_intra_auctions.append([])
    
    for i in range(len(Bidders)):   # For every bidder
        auctions = Bidders[i][1];
        for j in range(len(auctions)):   # For every Auction
            times_a = auctions[j][1];
            time_sub = get_difference_array(times_a)/(np.power(10,9))
            # Append the whole list since we are working with the same bidder
            own_time_distance_previous_intra_auctions[i].extend(time_sub)
    return own_time_distance_previous_intra_auctions



def get_own_time_distance_previous_inter_auctions(Bidders,Auctions):
    own_time_distance_previous_inter_auctions = []
    for i in range(len(Bidders)):   # For every bidder
        all_bids = []
        auctions = Bidders[i][1];
        for j in range(len(auctions)):   # Join all the bids from an auction
            all_bids.extend(auctions[j][1])
            
        all_bids = np.array(all_bids)
        all_bids = np.sort(all_bids)       # Sort them
        
        all_bids_sub = get_difference_array(all_bids)/(np.power(10,9))
        
        own_time_distance_previous_inter_auctions.append(all_bids_sub)
    return own_time_distance_previous_inter_auctions

def get_own_times_bidding(Bidders,Auctions):
    own_times_bidding = []
    for i in range(len(Bidders)):   # For every bidder
        all_bids = []
        auctions = Bidders[i][1];
        for j in range(len(auctions)):   # Join all the bids from an auction
            all_bids.extend(auctions[j][1])
            
        all_bids = np.array(all_bids)
        all_bids = np.sort(all_bids)/(np.power(10,9))    # Sort them
        
        own_times_bidding.append(all_bids)
    return own_times_bidding

def get_mybid_lastbid_distance(Bidders,Auctions):
    mybid_lastbid_distance = []
        
    for i in range(len(Bidders)):   # For every bidder
        auctions = Bidders[i][1];
        mybid_lastbid_distance.append([])  # Add bidder
        
        for j in range(len(auctions)):   # For every auction
            auction = auctions[j][0]     # auction
            mytime = auctions[j][1][-1]  # Time of my last bid
            lasttime = Auctions[auction][2][-1] # Last bid time in the auction
            distance = np.array((lasttime - mytime), dtype = float)/(np.power(10,9))
            mybid_lastbid_distance[i].append(distance)
    return mybid_lastbid_distance


def get_Auctions_timeType(Auctions):
    # We want to know for evert Auction, if it belongs to the one that 
    # finish in the 2nd column or in the 3rd 
    
    first_gap_center = 9.67 * np.power(10,15)  # More or less, a ojo
    second_gap_center = 9.73 * np.power(10,15)
    
    Auction_timeType = np.zeros(len(Auctions))
        
    for i in range(len(Auctions)):   # For every bidder
        min_value = np.min(Auctions[i][2])
        max_value = np.min(Auctions[i][2])
        
        if (max_value > second_gap_center): # If the last bid was made in the thir part
            Auction_timeType[i] = 1            # If it was not, we assume it was in the first.
            
    return Auction_timeType
    
###############################################################3
def num_bids_perAuction(Auctions, first_gap_center, second_gap_center):
    # We want to know for evert Auction, if it belongs to the one that 
    # finish in the 2nd column or in the 3rd 
    
    Auction_timeType = np.zeros(len(Auctions))
        
    for i in range(len(Auctions)):   # For every bidder
        min_value = np.min(Auctions[i][2])
        max_value = np.min(Auctions[i][2])
        
        if (max_value > second_gap_center): # If the last bid was made in the thir part
            Auction_timeType[i] = 1            # If it was not, we assume it was in the first.
            
    return Auction_timeType
    
def get_Bidders_timeType(Bidders, first_gap_center, second_gap_center):
    # We want to know for every Bidder, the proportion of bids it makes in the first
    # two columns and the proportion in the third column
    # AT THE END, all values are 1 or 0 which means that one user only bids in one of the sides.
    # Which means we @@@@@@ CAN TOTALLY SEPARATE THE THIRD COLUMN FORM THE OTHER 2 @@@@@
    
    Bidder_timeType = np.zeros(len(Bidders))
        
    for i in range(len(Bidders)):
        auctions = Bidders[i][1]
        for j in range(len(auctions)): 
            bids = auctions[j][1]
            min_value = np.min(bids)
            max_value = np.min(bids)
            
            if (max_value > second_gap_center): # If the last bid was made in the thir part
                Bidder_timeType[i] += 1            # If it was not, we assume it was in the first.
        if (auctions != []):
            Bidder_timeType[i] /= len(auctions)
        
    return Bidder_timeType
    
def get_timeGapsLength(Auctions, first_gap_center, second_gap_center):
    # We want to know for evert Auction, if it belongs to the one that 
    # finish in the 2nd column or in the 3rd in order to know how to  remove the times
    
    First_column_max = []
    Second_column_min = []
    
    Second_column_max = []
    Third_column_min = []
    
    First_column_num = 0;   # These turn to be the number of auctions where they appear. The 1 and 2 have overlapping auctions.
    Second_column_num = 0;
    Third_column_num = 0;
    
    for i in range(len(Auctions)):   # For every bidder
        times_a = np.array(Auctions[i][2])
#        print len(times_a)
        
        # First Column
        indexes = np.where(times_a < first_gap_center)
#        print len(indexes[0])
        if (indexes[0] != []):
            First_column_num += len(indexes[0])
            aux = np.max(times_a[indexes])
            First_column_max.append(aux)
         
        # Second Column
        indexes = np.where((times_a > first_gap_center))
        aux = times_a[indexes]
        if (indexes[0] != []):
            indexes = np.where(aux < second_gap_center)
#            print len(indexes[0])
            if (indexes[0] != []):
                Second_column_num += len(indexes[0])
                Second_column_min.append(np.min(aux))
                Second_column_max.append(np.max(aux))
                
        # Third Columns    
        indexes = np.where(times_a > second_gap_center)
#        print len(indexes[0])
        if (indexes[0] != []):
            Third_column_num += len(indexes[0])
            aux = np.min(times_a[indexes])
            Third_column_min.append(aux)
        

    
#    print Second_column_min
#    print First_column_max
#    print Third_column_min
#    print Second_column_max
    
    Second_column_min = np.array(Second_column_min)    
    First_column_max = np.array(First_column_max) 
    Third_column_min = np.array(Third_column_min)    
    Second_column_max = np.array(Second_column_max) 
    
    # Get the gaps length
    first_gap_length = np.min(Second_column_min) - np.max(First_column_max)
    second_gap_length = np.min(Third_column_min) - np.max(Second_column_max)
    gaps_length = [first_gap_length,second_gap_length]
    
    # Get the Time positions of the second column
    Second_column_pos = [np.min(Second_column_min),np.max(Second_column_max)];  # Same for the 2 columns
    
    # Get the number of bids in each column    
    num_bids_column = (First_column_num,Second_column_num,Third_column_num)

    return (gaps_length, Second_column_pos, num_bids_column)



def remove_time_gaps(Auctions, Bidders,first_gap_center, second_gap_center, time1, time2):

#    time1,time2,nums, Data_time_length = get_timeGapsLength(Auctions)

    
    # Remove it from the Auctions structure
    for i in range(len(Auctions)):
        time_a = Auctions[i][2]    # Maybe transform to np.array of float
    #    time_a = np.sort(time_a)
        for j in range(len(time_a)):
            if (time_a[j] > first_gap_center):
                    if (time_a[j] > second_gap_center):
                        time_a[j] -= time2
                    time_a[j] -= time1
    
    # Remove it from the Bidders structure
    for i in range(len(Bidders)):
        auctions = Bidders[i][1]
        for j in range(len(auctions)): 
            bids = auctions[j][1]
            for k in range (len(bids)): # For every bid
                if (bids[k] > first_gap_center):
                        if (bids[k] > second_gap_center):
                            bids[k] -= time2
                        bids[k] -= time1

def get_allBids_Distribution(Auctions, n_div):
# This function gets all the bids, ordered by time, and then obtains the N-div
# frecuency distribution to see how time has been changed !
    
    Total_bids = []
    for i in range(len(Auctions)):
        time_a = Auctions[i][2]    # Maybe transform to np.array of float
        Total_bids.extend(time_a)
    
    Total_bids.sort()    # Sort ascending order:
    
    min_value = Total_bids[0]
    max_value = Total_bids[-1]
    
    probs = np.zeros(n_div);
    
    current_div = 0;
    for b in Total_bids:
        if (b > (min_value + (max_value - min_value)*(current_div + 1)/n_div)):
            current_div += 1
        probs[current_div] += 1
    
    return probs
        
    
def get_Bids_Distribution(Bids, n_div):
# This function gets all the bids, ordered by time, and then obtains the N-div
# frecuency distribution to see how time has been changed !
    
    Total_bids = Bids

    Total_bids.sort()    # Sort ascending order:
    
    min_value = Total_bids[0]
    max_value = Total_bids[-1]
    
    probs = np.zeros(n_div);
    
    current_div = 0;
    for b in Total_bids:
        if (b > (min_value + (max_value - min_value)*(current_div + 1)/n_div)):
            current_div += 1
        probs[current_div] += 1
    
    return probs/len(Total_bids)     
    


def plot_Auctions_bots_less100 (very_human_bot_indx, Auctions, Bidders):
    for i in range (len(very_human_bot_indx)):  # Get their type of good
        
        times_a = Auctions[Bidders[very_human_bot_indx[i]][1][0][0]][2]
        pl.scatter(times_a,0.1*i*np.ones((1,len(times_a))),color="blue") # Plot auctions
        last_bid = Bidders[very_human_bot_indx[i]][1][0][1][0]
        pl.scatter(last_bid,0.1*i,color="red")
