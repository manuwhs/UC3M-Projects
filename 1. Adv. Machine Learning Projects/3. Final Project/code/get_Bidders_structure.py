
# coding: utf-8

# In[1]:

import pandas as pd
import utilities_2 as util2
import time
import pickle
import gc

# Load data

load_data = 0  # Flag to load the data
Bidders = []  # List with the auction structures. 

if (load_data == 1):
    print "Loading Data"
    bids = pd.read_csv('data/bids/bids.csv')
    n_bids, n_attr = bids.shape   # Get the number of bits and attr
    
    # Sort DataFrame by auction number
    print "Sorting bids by auction"
    # Order first by bidder and then by auction(indise each bidder)
    bids.sort(['bidder_id','auction'], ascending=[1, 1], axis = 0, inplace = True) # Inplace so that it changes the current, not create a new one
    bids.index = range(n_bids) # For some reason the "index" of the table is not changed by sort, so it has to be changed.
    # Now that they are sorted, it is easier and faster to create our structure. 
    # We just have to see if the current auction is different from the next so that
    # we have to change it.
    #==============================================================================
    # # First make a search of the boundaries of each user:
    #==============================================================================
    print "Getting the limits of the bidders in the dataset"
    bidders_start = [];

    for r in range(n_bids):
        current_bidder = bids['bidder_id'][r]
        if (current_bidder != last_bidder):  # If we are in the same bidder
            bidders_start.append(r)
        last_bidder = current_bidder

    n_bidders = len(bidders_start)
    bidders_start.append(n_bids)   # Limit of the last bidder
#==============================================================================
# CREATE AUCTIONS STRUCTURE
#==============================================================================
n_bids, n_attr = bids.shape   # Get the number of bits and attr
n_bidder = 0;               # Number of bidders

# If the list cannot be ordered according to "auction" coz auction is a string
# and "sort" does not work with strings...


 
# #==============================================================================
# # 	· [2] Crear estructura por usuarios.
# # 
# # 		Por cada usuario, poner array objectos en los que ha pujado
# # 			Por cada objecto poner:
# # 				- Identificador del objeto
# # 				- Tipo 
# # 				- Si se gano o no
# # 				- Pujas. (Array) 
# # 				Por cada puja (ordenadas temporalmente) poner:
# # 					- Tiempo en el que se pujo
# # 					- Cantidad pujada
# # 					- Localizacion, IP
# #==============================================================================
# 
# #==============================================================================
# # # For that, we serach through the "bids" Dataframe, obtaining all the bids with 
# # # the same "auction" value (also eliminate them from the whole structe when readint them)
# # # Search process is as follows: 
# #     - Search the bids, position by position. For every row of the table
# #     - Get the "auction" and compare it with the ones already obtained.
# #     - Add it to the corresponding auction list. if it disnt exists create list.
#==============================================================================
#     - Every bid of this structure will have the proerties: 
#         - Who made the bid
#         - Time of the bid
#==============================================================================
        