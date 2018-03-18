
# coding: utf-8

import pandas as pd
import utilities_2 as util2
import time
import gc    # Carbage collector python.
import numpy as np

# Load data

load_data = 1  # Flag to load the data
free_mem = 0  #  Flag to delete all the data from RAM after writting file

# We want to reference every object and every user by a number (index).
# For that we do a preprocess stage where we asign these values.

# What we do is: We sort by auction, we change names by numbers, then we sort by bidder
# and we change by numbers


    
if (load_data == 1):
    print "Loading Data"
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    bids = pd.read_csv('data/bids/bids.csv')
    n_bids, n_attr = bids.shape   # Get the number of bits and attr
    
    # Give values the limits of acutions and bidders
    print "Obtaining the indexes of Auctions"
    auctions_start, auctions_id  = util2.getadd_auctions_indx(bids)
    print "Obtaining the indexes of Bidders"
    bidders_start, bidders_id = util2.getadd_bidders_indx_starts(bids)
    

    # Now that they are sorted, it is easier and faster to create our structure. 
    # We just have to see if the current auction is different from the next so that
    # we have to change it.
    
#==============================================================================
# CREATE AUCTIONS STRUCTURE
#==============================================================================
print "Getting Auction Structure"
Auctions = util2.get_Auctions_structure(bids,auctions_start, "yes")
del Auctions
gc.collect()

#==============================================================================
# CREATE BIDDERS STRUCTURE
#==============================================================================
# Sort again by time to reestart

print "Getting Bidder Structure"
Bidders = util2.get_Bidders_structure(bids,bidders_start, "yes")
del Bidders
gc.collect()

# Store the ID´s !!
util2.store_pickle ("Auction_id", auctions_id, 1)
util2.store_pickle ("Bidder_id", bidders_id, 1)


# pickle uses too much RAM memory due to how it is internally implemented, We could
# try some other aproach like h5file but probably, we will have to implement something
# to read it as well

#==============================================================================
# # We want to create a numpy structure where:
# 	· [1] Crear estructura por objetos.
# 		Por cada objeto:
# 			- Identificador del objeto
# 			- Tipo 
# 			- Array de Instantes de Pujas
#          - Array de los "bidder_id" de los que pujaron
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
#         with open(filename + str(partitions)+ ".pkl", 'wb') as f:
#         pickle.dump(len(li), f)    # We dump the auctions one by one coz pickle uses a lot of memmory
#         for a in Auctions:
#             pickle.dump(a, f)
#==============================================================================
