
import pickle
import numpy as np
import pandas as pd
import gc

def store_pickle (filename, li, partitions = 1):
    gc.collect()
    # This function stores the list i into a number of files equal to partitions in pickle format
    num = int(len(li)/partitions);
    
    for i in range(partitions - 1):
        print "Creating file: " + filename + str(i)+ ".pkl"
        with open(filename + str(i)+ ".pkl", 'wb') as f:
            pickle.dump(li[i*num:(i+1)*num], f)    # We dump the auctions one by one coz pickle uses a lot of memmory
    
    print "Creating file: " + filename + str(partitions -1)+ ".pkl"
    with open(filename + str(partitions - 1)+ ".pkl", 'wb') as f:
            pickle.dump(li[num*(partitions - 1):], f)    # We dump the auctions one by one coz pickle uses a lot of memmory
    gc.collect()
    
def load_pickle (filename, partitions = 1):
    gc.collect()
    total_list = []
    for i in range(partitions):
        print "Loading file: " + filename + str(i)+ ".pkl"
        with open(filename + str(i)+ ".pkl", 'rb') as f:
            part = pickle.load(f)    # We dump the auctions one by one coz pickle uses a lot of memmory
        total_list.extend(part)
        
    gc.collect()
    return total_list



def getadd_auctions_indx (bids):
    gc.collect()
    n_bids, n_attr = bids.shape   # Get the number of bits and attr
    bids.sort(['auction'], ascending=[1], axis = 0, inplace = True)
    bids.index = range(n_bids) # For some reason the "index" of the table is not changed by sort, so it has to be changed.
    
    auc_indx = np.empty(n_bids, dtype = 'int32')
    last_auction = "NOT A REAL AUCTION"
    # Get the intervals
    n = -1;
    auctions_start = []
    for r in range(n_bids):
         current_auction = bids['auction'][r]
         if (current_auction != last_auction):  # If we arent in the same auction
             auctions_start.append(r)           # The first time will go here
             n += 1;
         last_auction = current_auction
         auc_indx[r] = n
    auctions_start.append(n_bids)    # Limit of the last auction
    
    bids['auction_indx'] = pd.Series( auc_indx, index = bids.index) # Add name in int form to

    
    # Get the array of names of the auctions:
    auctions_id = []
    for i in range(len(auctions_start)-1):
        auctions_id.append(bids.auction[auctions_start[i]])
        
    bids.drop('auction', axis=1, inplace=True)  # Eliminate auction
    return (auctions_start, auctions_id)

def getadd_bidders_indx_starts (bids):
    gc.collect()
    n_bids, n_attr = bids.shape   # Get the number of bits and attr
    bids.sort(['bidder_id'], ascending=[1], axis = 0, inplace = True)
    bids.index = range(n_bids) # For some reason the "index" of the table is not changed by sort, so it has to be changed.  
   
    bidder_indx = np.empty(n_bids,  dtype = 'int32')
    last_bidder = "NOT A REAL AUCTION"
    # Get the intervals
    n = -1;
    bidders_start = []
    for r in range(n_bids):
         current_bidder = bids['bidder_id'][r]
         if (current_bidder != last_bidder):  # If we arent in the same bidder
             bidders_start.append(r)         # The first time will go here
             n += 1;
         last_bidder = current_bidder
         bidder_indx[r] = n
    bidders_start.append(n_bids)    # Limit of the last auction
    
    bids['bidder_indx'] = pd.Series( bidder_indx, index = bids.index) # Add name in int form to

    # Get the array of names of the bidders:
    bidders_id = []
    
    for i in range(len(bidders_start)-1):
        bidders_id.append(bids.bidder_id[bidders_start[i]])
    bids.drop('bidder_id', axis=1, inplace=True)  # Eliminate bidder
    return (bidders_start, bidders_id)
    
    
def get_Auctions_structure(bids,auctions_start, output_file):
    # This function required bids to be sorted by time
    # Sort by Auction
    gc.collect()
    n_bids, n_attr = bids.shape   # Get the number of bits and attr
    print "Sorting by Auctions"
    bids.sort(['auction_indx', 'time'], ascending=[1, 1], axis = 0, inplace = True)
    bids.index = range(n_bids) # For some reason the "index" of the table is not changed by sort, so it has to be changed.
    gc.collect()
    
    n_auction = 0;             # Auction number
    Auctions = []  # List with the auction structures. 
    last_auction = "Not a Real Auction"
    print "Getting the structure of Auctions"
    for i in range(n_bids):     # n_bids
        current_auction = bids['auction_indx'][i]
        if (current_auction != last_auction):  # If we are in the same auction
            print current_auction
            n_auction += 1      # Increase the number of auctions
            auction = (bids['auction_indx'][i],bids['merchandise'][i], [], []) # Create main auction data
            Auctions.append(auction)       # Include new auction into structure
                                         # First array is time, second is users
        bid_aux = (bids['time'][i], bids['bidder_indx'][i])  # Create specific bid data of that auction
        Auctions[n_auction-1][2].append(bid_aux[0])   # Include new bid time into auction structure
        Auctions[n_auction-1][3].append(bid_aux[1])  # Include new bidder_id time into auction structure
       
        last_auction = current_auction

    if (output_file == "yes"):
        print "Writting to file"
        store_pickle ("Auctions", Auctions, 5)
    print "Done !!"
    return Auctions
    
def get_Bidders_structure(bids,bidders_start, output_file):
    # This function required bids to be sorted by time
    # Sort by Auction
    print " Sorting by Bidders and Auctions"
    n_bids, n_attr = bids.shape   # Get the number of bits and attr
    bids.sort(['bidder_indx','auction_indx', 'time'], ascending=[1, 1, 1], axis = 0, inplace = True) # Inplace so that it changes the current, not create a new one
    bids.index = range(n_bids) # For some reason the "index" of the table is not changed by sort, so it has to be changed.
    
    Bidders = []  # List with the auction structures. 
    last_auction = "NOT A REAL AUCTION"
    n_bidders = len(bidders_start) - 1;               # Number of bidders
     #==============================================================================
     # FOR EVERY BIDDER   n_bidders
     #==============================================================================
    print "Getting Bidder Structure"
    for b in range(n_bidders):   # For every bidder  n_bidders
         print b
         ini = bidders_start[b]
         fin = bidders_start[b+1]
         
         # Bidder has:  Bidder_id, Bids Time Array, Countries Array, IPs Array, Mobiles Array, URL Array
         bidder = (bids['bidder_indx'][ini],[],[],[],[])
         Bidders.append(bidder)          # Append bidders
         
         # Get the Auction Limits (ordered by time each auction)
         auctions_start = []
         for r in range(ini, fin):
             current_auction = bids['auction_indx'][r]
             if (current_auction != last_auction):  # If we are in the same bidder
                 auctions_start.append(r)
             last_auction = current_auction
             
         n_auctions = len(auctions_start);  # Number of auctions of the bidder
         auctions_start.append(fin)    # Limit of the last auction
     #==============================================================================
     # FOR EVERY AUCTION OF THAT BIDDER
     #==============================================================================
     
         for a in range (n_auctions):
             ini_a = auctions_start[a]
             fin_a = auctions_start[a+1]
             auction = (bids['auction_indx'][ini_a], [] ) # Create main auction data 
             Bidders[b][1].append(auction)          # Append object into bidders array of acutions
     
     #==============================================================================
     # FOR EVERY BID INSIDE THAT AUCTION
     #==============================================================================
             for r in range(ini_a, fin_a):
                 Bidders[b][1][a][1].append(bids['time'][r])     # Append bid time and shit
                 # Here we could check for the bidders IP, URL and Country to see if they change.
               
     #==============================================================================
     #           # For checking everything is correct
     #           if (bids['bidder_id'][r] != Bidders[b][0]):
     #                print "Gilipollas"
     #==============================================================================
                    
#==============================================================================
#  THIS METHODOLOGY IS VERY SLOW !!! MAYBE CREATING THE WHOLE ARRAY AND 
# AT THE END USE "UNIQUE" TO GET THE unique ones
#==============================================================================
                # IP finding
                 Bidders[b][2].append(bids['ip'][r])
                  
                # Country finding 
                 Bidders[b][3].append(bids['country'][r])
                  
                # Mobile finding 
                 Bidders[b][4].append(bids['device'][r])      

     #==============================================================================
     # for i in range (n_bidder): # Transform time instances into np array
     #     Bidders[i][2][0] = np.array(Bidders[i][2][0]) 
     #==============================================================================
             
     # USE pickle library to export it into a file, and read it afterwards
    if (output_file == "yes"):
        print "Writting to file"
        store_pickle ("Bidders", Bidders, 5)
    return Bidders
    
    
    
    
    
    
    
#==============================================================================
#                  IP = bids['ip'][r]
#                  match = 0;
#                  for ip_i in Bidders[b][2]:  # The 2 is the IPs array position
#                      if (ip_i == IP):  # If the IP is already in the array
#                          match = 1;
#                          break
#                  if (match == 0):
#                      Bidders[b][2].append(IP)
#==============================================================================
    
    
    
    
    
    