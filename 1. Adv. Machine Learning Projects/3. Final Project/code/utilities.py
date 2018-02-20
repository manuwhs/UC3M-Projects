import pandas as pd
import numpy as np

def selectBidsByUser(bids,user):
    return bids[bids['bidder_id']==user]

def selectBidsByAuction(bids,auction):
    return bids[bids['auction']==auction]