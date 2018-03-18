
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import scipy as sp
import utilities as util
import time
import pickle

def chars2int ( chars ):
    number = 0;
    for i in range(len(chars)):
        number += ord(chars[i]) << i*8
    return number

def int2chars ( number ):
    chars = '';
    for i in range(len(chars)):
        number >> 8;
        if (number > 0):
            chars += chr(chars[i] & 255) 
        else:
            break
    
    return chars

# In[2]:

# Load data

load_data = 1  # Flag to load the data

n_bids, n_attr = bids.shape   # Get the number of bits and attr
Auctions = []  # List with the auction structures. 

if (load_data == 0):
    print "Loading Data"
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    bids = pd.read_csv('data/bids/bids.csv')

    # Sort DataFrame by auction number
    print "Sorting bids by auction"
    bids.sort(['auction'], ascending=[1], axis = 0, inplace = True) # Inplace so that it changes the current, not create a new one
    bids.index = range(n_bids) # For some reason the "index" of the table is not changed by sort, so it has to be changed.
    # Now that they are sorted, it is easier and faster to create our structure. 
    # We just have to see if the current auction is different from the next so that
    # we have to change it.
    





#==============================================================================
# # We want to create a numpy structure where:
# 	· [1] Crear estructura por objetos.
# 		Por cada objeto:
# 			- Identificador del objeto
# 			- Tipo 
# 			- Pujas. (Array) 
# 				Por cada puja (ordenadas temporalmente) poner:
# 					- Quien pujo (identificador, IP, localizacion...)
# 					- Tiempo en el que se pujo
# 					- Cantidad pujada
#==============================================================================


#==============================================================================
# # For that, we serach through the "bids" Dataframe, obtaining all the bids with 
# # the same "auction" value (also eliminate them from the whole structe when readint them)
# # Search process is as follows: 
#     - Search the bids, position by position. For every row of the table
#     - Get the "auction" and compare it with the ones already obtained.
#     - Add it to the corresponding auction list. if it disnt exists create list.
#     - Every bid of this structure will have the proerties: 
#         - Who made the bid
#         - Time of the bid
#==============================================================================
        

#==============================================================================
# CREATE AUCTIONS STRUCTURE
#==============================================================================

n_auction = 0;             # Auction number
bid_p = 0;                  # Bid indx to go through the bids array

# If the list cannot be ordered according to "auction" coz auction is a string
# and "sort" does not work with strings...


last_auction = "NOT A REAL AUCTION"

while (bid_p < 300000):
    current_auction = bids['auction'][bid_p]
    
    if (current_auction == last_auction):  # If we are in the same auction
#        print current_auction + next_auction
        bid_aux = (bids['time'][bid_p], bids['bidder_id'][bid_p])  # Create specific bid data of that auction
        Auctions[n_auction-1][2].append(bid_aux)   # Include new bid into auction structure

    else:                   # If we change auction 
        n_auction += 1      # Increase the number of auctions
        auction = (bids['auction'][bid_p],bids['merchandise'][bid_p], [] ) # Create main auction data
        bid_aux = (bids['time'][bid_p], bids['bidder_id'][bid_p])  # Create specific bid data of that auction
        
        
        Auctions.append(auction)       # Include new auction into structure
        Auctions[n_auction-1][2].append(bid_aux)   # Include new bid into auction structure

    bid_p += 1
    last_auction = current_auction

for i in range (n_auction): # Transform time instances into np array
    Auctions[i][2][0] = np.array(Auctions[i][2][0]) 
# USE pickle library to export it into a file, and read it afterwards
with open("Auctions"+ ".pkl", 'wb') as f:
    pickle.dump(Auctions, f)





# We have to transform bids.auction to numeric. Since the auctions are not in
# decimal or hexadecimal, we have to transform them first into numbers and then
# perform the sorting

#==============================================================================
# print "Transformin 'Char array' into int"
# auction_numbers = map (chars2int,bids.auction)  # Transform names into "int"
# bids['auction_number'] = pd.Series( auction_numbers, index = bids.index) # Add name in int form to the dataframe
# n_attr += 1;
#==============================================================================





# For when the list is not ordered
#while (bid_p < 100000):
#    current_bid = bids['auction'][bid_p]  # Get the auction
#    
#    new_auc = 1;
#    # If the auction already exists:
#    for i in range (n_auction):
#        if (current_bid == Auctions[i][0]):
#            bid_aux = (bids['time'][bid_p], bids['bidder_id'][bid_p])  # Create specific bid data of that auction
#            Auctions[i][2].append(bid_aux)   # Include bid
#            new_auc = 0;
#            break
#        # If we already had the auction the break would get us 
#    if (new_auc == 1):
#        auction = (bids['auction'][bid_p],bids['merchandise'][bid_p], [] ) # Create main auction data
#        bid_aux = (bids['time'][bid_p], bids['bidder_id'][bid_p])  # Create specific bid data of that auction
#    
#        Auctions.append(auction)       # Include new auction into structure
#        Auctions[n_auction][2].append(bid_aux)   # Include new bid into auction structure
#        n_auction += 1      # Increase the number of auctions
#    
#    bid_p += 1
    