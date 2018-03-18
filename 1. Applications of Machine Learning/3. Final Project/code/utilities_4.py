
import pickle
import numpy as np
import matplotlib as plt
import pylab as pl
import utilities_3 as util3
# Utilities for obtaining features:
#    - From time vectors
#    - Or from IPs, countries....
#    - Or from anything really...


def get_number_of_Bids(Bidders):
    number_of_bids = np.zeros(len(Bidders))
    for i in range (len(Bidders)):          # For every user
        for a in range(len(Bidders[i][1])): # For every auction of the user
            number_of_bids[i] += len(Bidders[i][1][a][1])  # Numpy array
    return number_of_bids
            
def get_number_of_Auctions(Bidders):
    number_of_auctions = np.zeros(len(Bidders))
    for i in range (len(Bidders)):          # For every user
        number_of_auctions[i] += (len(Bidders[i][1]))
    return number_of_auctions

def get_number_of_Auctions_won(Auctions,Bidders):
    number_of_auctions_won = np.zeros(len(Bidders))
    for i in range (len(Auctions)):          # For every auction
        number_of_auctions_won[Auctions[i][3][-1]] += 1 # Get the user that made the last bid
    return number_of_auctions_won

def get_proportion_consecutive_bids(Auctions,Bidders):
    proportion_consecutive_bids = np.zeros(len(Bidders))
    for i in range (len(Auctions)):          # For every auction
        bids = Auctions[i][2]
        
        last_bidder = Auctions[i][3][0]
        for b in range (1,len(bids)):
            new_bidder = Auctions[i][3][b]
            if (last_bidder == new_bidder):
                proportion_consecutive_bids[last_bidder] += 1;
            
            last_bidder = new_bidder
   
    number_of_Bids = get_number_of_Bids(Bidders)
    for i in range (len(Bidders)):          # For every user
        if (number_of_Bids[i] > 0):
            proportion_consecutive_bids[i] /= number_of_Bids[i]
        
    return proportion_consecutive_bids
    

def plot_2_features_train(train, Bidders_dic, fx, fy):
    for i in range (len(train)):
        print i
        indx = Bidders_dic[train.bidder_id[i]]
#        print i
        pl.figure("2 features")
        if (train.outcome[i] < 0.5):  # Outcome of the i-th bidder in train
            pl.scatter(fx[indx],fy[indx],color="blue")
        else:
            pl.scatter(fx[indx],fy[indx],color="red")


def get_bot_number(bidders,Bidders_dic):
    n_bots = 0;
    for i in range (len(bidders)):
        if (bidders.outcome[i] > 0.5): 
            n_bots += 1;
            
    return n_bots

def get_AuctionsLength(Auctions):
    AuctionsLength = np.zeros(len(Auctions))
    for i in range (len(Auctions)):
        AuctionsLength[i] = Auctions[i][2][-1] - Auctions[i][2][0]
    return AuctionsLength
    
def get_AuctionsEnd(Auctions):
    get_AuctionsEnd = np.zeros(len(Auctions))
    for i in range (len(Auctions)):
        get_AuctionsEnd[i] = Auctions[i][2][-1]
    return get_AuctionsEnd

def get_AuctionsStart(Auctions):
    get_AuctionsStart = np.zeros(len(Auctions))
    for i in range (len(Auctions)):
        get_AuctionsStart[i] = Auctions[i][2][0]
    return get_AuctionsStart
    
#==============================================================================
# Puerta de Toledo, Almudena, Palacio Real, Calle Mayor, Parar en plaza Mayor, 
# Continuar hasta Sol, De Sol coger Preciados para subir a Callao, bajas gran via
# hasta Montera, bajas hasta el edificio de metropolis (inicio de la gran via).
# Bajar a cibeles, bajar por paseo del prado hasta atocha. 
#==============================================================================

def get_meanVarAndProbs_by_ranges (Time_features, n_div, type_div):
    num_bidders = len(Time_features)
    probs_list = []
    means_list = []
    vars_list = []
    
    for i in range (n_div):
        means_list.append(np.zeros(num_bidders))
        probs_list.append(np.zeros(num_bidders))
        vars_list.append(np.zeros(num_bidders))
        
    for i in range (num_bidders):
        times = np.array(Time_features[i])
        print i
        if (len(times) > 2):
            times.sort()
            min_t = np.min(times)
            max_t = np.max(times)
            
            probs = np.zeros(n_div);
            
            current_div = 0;
            
            list_items = []
            
            for t in times:
                if (t > (min_t + (max_t - min_t)*(current_div + 1)/n_div) + 0.001):
                    # the 0.0001 is to avoid float "bit-step" errors
                    means_list[current_div][i] = np.mean(np.array(list_items))    
                    current_div += 1
                    list_items = []
                    
                probs[current_div] += 1
                list_items.append(t)
                
            means_list[current_div][i] = np.mean(np.array(list_items)) 
            vars_list[current_div][i] = np.var(np.array(list_items)) 
            probs = probs/len(times)
            
            for j in range (n_div):
                probs_list[j][i] = probs[j]
    
    return (probs_list,means_list, vars_list)
        
def get_final_features (Auctions, Bidders):
    number_of_bids = util3.get_number_of_Bids(Bidders)
    number_of_auctions = util3.get_number_of_Auctions(Bidders)
    # This could be removed if they didnt actually give us this info
    number_of_auctions_won = util3.get_number_of_Auctions_won(Auctions,Bidders)
  
    # array of the instances where the users bid
    own_times_bidding = util3.get_own_times_bidding(Bidders,Auctions);
    
    # Time distances between the moment I bid and the last bid instance
    time_distance_previous = util3.get_time_distance_previous (Bidders,Auctions);
    
    # Time distances between 2 consecutive own bids inside an auction
    own_time_distance_previous_intra_auctions = util3.get_own_time_distance_previous_intra_auctions(Bidders,Auctions)
    
    # Time distances between 2 consecutive own bids in any auction
    own_time_distance_previous_inter_auctions = util3.get_own_time_distance_previous_inter_auctions(Bidders,Auctions)
    
    # Distance from my last bid to the last bid
    mybid_lastbid_distance = util3.get_mybid_lastbid_distance(Bidders,Auctions)
    
    