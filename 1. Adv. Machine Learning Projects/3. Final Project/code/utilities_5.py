import numpy as np
import pandas as pd

def system_random (features, Bidders_dic):
    bidders = pd.read_csv('data/test.csv')
    results = np.zeros(len(bidders))
    for i in range (len(bidders)):
        
        try:
            indx = Bidders_dic[bidders.bidder_id[i]]
            print indx
            print features[indx]
            
            if (features[indx] < 120 ):
                results[i] = 0;
            else:
                results[i] = np.random.uniform(0,0.6)
            
        except KeyError:
            results[i] = 0
            
    return results


def system_trained (features, Bidders_dic, test_selection):
    bidders = pd.read_csv('data/test.csv')
    results = np.zeros(len(bidders))
    for i in range (len(bidders)):
        
        try:
            indx = Bidders_dic[bidders.bidder_id[i]]
            print indx
            print features[indx]
            
            if (features[indx] < 120 ):
                results[i] = 0;
            else:
                results[i] = np.random.uniform(0,0.6)
            
        except KeyError:
            results[i] = 0
            
    return results
    
    
def write_csv_output( results, filename):
    bidders = pd.read_csv('data/test.csv')
    with open(filename, 'wb') as fd:
        fd.write ("bidder_id,prediction\n")
        for i in range (len(bidders)):
            bider_name = str(bidders.bidder_id[i])
            result = str(results[i])
            fd.write( bider_name+ "," + result + "\n")
            
        