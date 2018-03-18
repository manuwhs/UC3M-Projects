
import numpy as np

#==============================================================================
# TEXT PREPROCESSING Functions
#==============================================================================
# Tratment of the words of the document
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

def doc_preprocess(document, mode = 0):
    # Preprocesees a document:  Tokenization, lemmatization...
    if (mode == 1):print " "; print document; print " "
    document = doc_tokeniz(document) 
    if (mode == 1):print document; print " "
    document = doc_lowercase(document)
    if (mode == 1):print document; print " "
    document = doc_rem_punctuation(document) 
    if (mode == 1):print document; print " "
    document = doc_rem_stopwords(document) 
    if (mode == 1):print document; print " "
    document = doc_stem(document) 
    if (mode == 1):print document; print " "
    return document

def doc_tokeniz(document):
    # Gets the tokens of the document. The tokens are words.
    tokens = word_tokenize(document) 
    return tokens
    
def doc_lowercase (document):
    # Transforms a list of words (document) into lowercase
    low_text = [w.lower() for w in document] 
    return low_text
    
def doc_rem_stopwords(document):
    # Removes stopwords obtained from the nltk english corpus.
    stopwords_en = stopwords.words('english')
    clean_text = [word for word in document if not word in stopwords_en]
    return clean_text
    
def doc_stem(document):
    # Performs the english steaming of the words
    stemmer = SnowballStemmer('english')
    steammed_text = [stemmer.stem(word)for word in document]
    return steammed_text
    
def doc_rem_punctuation(document):
    # Removes punctuation marks
    clean_text = [w for w in document if w.isalnum()]
    return clean_text
    
#==============================================================================
# General functions
#==============================================================================
    
def total_words(document):
    return len(document)

#==============================================================================
# GET TF-IDF functions
#==============================================================================

def tf_idf_doc(document, doc_database):
    # This function outputs the tf-idf of a given document and the corresponding words
    # The "idf" function needs all of the database documents.
    
    [words, freqs] = get_words_freqs(document)
    
    tfs = get_tfs(freqs)
    idfs = get_idfs(words, doc_database)
    
    tfidfs = np.array(tfs*idfs)
    return [words, tfidfs]
    
  # ************** GET TFs  *************
    
def get_words_freqs(document):
    # Gets the frequency of every word in the document
    from nltk import FreqDist
    fdist = FreqDist(document)
    frecs = fdist.most_common()
    # Returns the words and frecuencies in tuples ordered.
    # frec = [('.', 5), ('me', 4), ('them', 4).....
    # We want to separate it into words and frecuencies
    N_words = total_words(frecs)
    freqs = np.empty(N_words)  # List of frecuencies
    words = []                # List of words
    
    for i in range (N_words):
        words.append(frecs[i][0])
        freqs[i] = int(frecs[i][1])
    return [words, freqs]
    
def get_tfs(freqs):
    # Gets the tfs using the frequency vector.
    # This functions assumes the frequencies are order in decreasing order
    # Some transformation could be applied to this function, thats why
    # we have made a function with such a simple initial content.
    return (freqs/float(freqs[0]))

 # ************** GET IDFs  *************

def get_idfs(document, doc_database):
    # Gets the idfs using the preprocessed words of the query and the dataset.

    N_doc = float(len(doc_database)) # Number of documents in database
    N_words = total_words(document)    # Number of words in the document
    num_docs = np.empty(N_words) # Here we save the number of documents containing every word.
    
    for i in range (N_words):   # For, every word get the number of documents containing it
        num_docs[i] = num_docs_containing(document[i], doc_database)
        
    return  np.log( N_doc / num_docs)


def num_docs_containing(word, doc_database):
    count = 0
#    print word
    for document in doc_database: # If the word is in the documen
#        print document[0]
        if (word in document[0]):
            count += 1
    return 1 + count   # We put minimum 1 so that this is not 0
 
 
#==============================================================================
# SIMILARITY functions
#==============================================================================

# Sort words to make posterior search easier and faster.
# TFIDF values are sorted accordingly. 
def doc_sort_words(words_tfidf):
    words = words_tfidf[0]
    tfidf = words_tfidf[1]
    
    together = zip(words, tfidf)  # Zip both arrays together
    sorted_together =  sorted(together) # Sorted using tfidf
    
    words = [x[0] for x in sorted_together]
    tfidf = [x[1] for x in sorted_together]
    
    return (words,np.array(tfidf))
     
 
# Define similarity values between the query and the corpus.
def get_similarities(query,database):
     query_words = doc_preprocess(query,1)  # Preprocess query
     query_words_tfidf = tf_idf_doc(query_words, database) # Get tfidf of query
     
     for i in range (len(query_words_tfidf[0])):
         print str(query_words_tfidf[1][i]) + "\t\t" + query_words_tfidf[0][i]
     print " "
     N_docs_database = len(database)
     
     eu_dists = np.zeros([N_docs_database,1])
     cos_sims = np.zeros([N_docs_database,1])
     n_commun = np.zeros([N_docs_database,1])
     tfidf_sum = np.zeros([N_docs_database,1])
     
     for i in range (N_docs_database):
         # Common words
         (vq ,vd) = get_common_tfidf_vectors(query_words_tfidf,database[i])
         # Non-common words
         (vnq ,vnd) = get_noncommon_tfidf_vectors(query_words_tfidf,database[i])
         
         # If no words in commun
         if(vq.size == 0):
              eu_dists[i] = -1
              cos_sims[i] = -1
              n_commun[i] = vq.size
         else:
             eu_dists[i] = get_euc_dist(vq,vd,vnq,vnd)
             cos_sims[i] = get_cos_sim(vq,vd)
             n_commun[i] = vq.size
             tfidf_sum[i] = np.sum(vd)
             
     similarities = (eu_dists,cos_sims, n_commun, tfidf_sum)
     # Similatiries obtained are:
     # - The euclidean distance
     # - The cosine similarity
     # - The number of words in commun. 
     #      (Important coz, they can have very high values in the eu y cos
     #      but just because they have very few words in commun)
     # - tfidf_sum: Sum of the tdifd of the values
     
     return similarities

def rank_documents(Similarity, N_top, type_simi):
    # RANK WITH EVERY INDEPENDENT SIMILARITY MEASURE.
    n_doc = len(Similarity)
    # 1) Euclidean distance: The closer, the better, so the smaller the value, the better
    if (type_simi == "euclidean"):
        # For this similarity, the vectors that did not have words in common have
        # similarity -1, so we cannot order them directly like this. 
        for i in range(len(Similarity)):
            if (Similarity[i] == -1):
                Similarity[i] = 10000000000;
        indexes = np.array(range(0,n_doc))
        together = zip(Similarity, indexes)  # Zip both arrays together
        sorted_together =  sorted(together) # Sorted Increasing order
        ordered_indexes = [x[1] for x in sorted_together]
        # Since ordered_indexes contains the indexes of the best documents 
        # (obtained from decreasing similarity)
        
    # 2) Cosine distance: The biger the value, the better 
    if (type_simi == "cosine"):
        # Wrong vectors have similarity -1 so no problem, the bigger the better.
        indexes = np.array(range(0,n_doc))
        together = zip(Similarity, indexes)  # Zip both arrays together
        sorted_together =  sorted(together, reverse=True) # Sorted Decreasing order
        ordered_indexes = [x[1] for x in sorted_together]
        #print ordered_indexes
    
    # 3) Use the number of words in common:
    if (type_simi == "common"):
        indexes = np.array(range(0,n_doc))
        together = zip(Similarity, indexes)  # Zip both arrays together
        sorted_together =  sorted(together, reverse=True) # Sorted Decreasing order
        ordered_indexes = [x[1] for x in sorted_together]
        
    # ) Use the sum of tfidf values:
    if (type_simi == "tdidf_sum"):
        indexes = np.array(range(0,n_doc))
        together = zip(Similarity, indexes)  # Zip both arrays together
        sorted_together =  sorted(together, reverse=True) # Sorted Decreasing order
        ordered_indexes = [x[1] for x in sorted_together]
        
    top_indexes = ordered_indexes[0:N_top]
    return top_indexes

def get_combined_rank(Ranks, N_top, type_comb):
# Ok... we have several similarity measures, now we have to output the ranking
# based on those similarities. How do we combine them to get the best ranking ?
    n_rank,n_doc = np.shape(Ranks)
    
    eu_w = 0.5; # Weight of the euclidean distance importance
    cos_w = 0.5; # Weight of the cosine distance importance
    n_w = 1;   # Weight of the nomber of common words importance
    
    total_similarity  = np.zeros([n_doc,1])
    
    for i in range(n_doc):
        total_similarity [Ranks[i]] += eu_w * Ranks[0][i] 
        total_similarity [Ranks[i]] += cos_w * Ranks[1][i] 
        total_similarity [Ranks[i]] -= n_w * Ranks[2][i] 
    
    # Get the top indexes rank from the combined similarity methods:
    indexes = np.array(range(0,n_doc))
    together = zip(total_similarity, indexes)  # Zip both arrays together
    sorted_together =  sorted(together) # Sorted Increasing order
    ordered_indexes = [x[1] for x in sorted_together]
    top_indexes = ordered_indexes[0:N_top]
    return top_indexes
    
def get_euc_dist(vq,vd,vnq,vnd):
    distance = np.sqrt(((vq - vd)*(vq - vd)).sum(axis=0))

    
    return distance
    
def get_cos_sim(vq,vd):
    modules = (np.sqrt((vq*vq).sum(axis=0)) * np.sqrt((vd*vd).sum(axis=0)))
    
    if (modules == 0):
        print "Error, cosine similarity. One vector is all 0s."
        return -1
    cos_sim = np.dot(vq.transpose(),vd)
    cos_sim = cos_sim / modules
    
    return cos_sim  
    

# Gets the commons words indexes between the string lists query and document
def get_common_words_index(query,document):
    # Given two lists of preprocessed words. It finds the common words 
    # among the 2 of them and returns the indexes of the words for both 
    # documents.
    N_query = total_words(query)
    
    query_indx = []
    document_indx = []
    
    for i in range(N_query):
        if (query[i] in document):
            position = document.index(query[i]) 
            query_indx.append(i)
            document_indx.append(position)
    
    return (np.array(query_indx, dtype=np.int64),np.array(document_indx, dtype=np.int64))


# Gets the diffetent words indexes between the string lists query and document
def get_noncommon_words_index(query,document,query_indx, document_indx):
    N_query = total_words(query)
    N_document = total_words(document)
    query_non_indx = range(N_query)
    document_non_indx = range(N_document)
    
    # Removes the commun indexes
    
    for i in range (len(query_indx)):
        query_non_indx.pop(query_indx[i]-i) 
    
    document_indx.sort()
    for i in range (len(document_indx)):
        document_non_indx.pop(document_indx[i]-i) 
    
    return (np.array(query_non_indx, dtype=np.int64),np.array(document_non_indx, dtype=np.int64))


# Gets the commons tfidf vectors between the query and the document
def get_common_tfidf_vectors(query_words_tfidf,document_words_tfidf):

    (query_indx,document_indx) = get_common_words_index(query_words_tfidf[0],document_words_tfidf[0])
    
#    print query_indx
#    print document_indx
    # Get the common (words_tfidf) vector of query aand documents
    query_common_tfidf = query_words_tfidf[1][query_indx]
    document_common_tfidf = document_words_tfidf[1][document_indx]
    
#    print query_words_tfidf
#    print document_words_tfidf
    
    return (query_common_tfidf,document_common_tfidf)

def get_noncommon_tfidf_vectors(query_words_tfidf,document_words_tfidf):

    (query_indx,document_indx) = get_common_words_index(query_words_tfidf[0],document_words_tfidf[0])
    (query_non_indx,document_non_indx) = get_noncommon_words_index(query_words_tfidf[0],document_words_tfidf[0],query_indx,document_indx )
#    print query_indx
#    print document_indx
    # Get the common (words_tfidf) vector of query aand documents
    query_noncommon_tfidf = query_words_tfidf[1][query_non_indx]
    document_noncommon_tfidf = document_words_tfidf[1][document_non_indx]
    
#    print query_words_tfidf
#    print document_words_tfidf
    
    return (query_noncommon_tfidf,document_noncommon_tfidf)


def rank_by_keywords(query_text, keydataset):
    # This just ranks the query according to the number of matching keywords of the
    # document. It assumes keydataset has already been preprocessed and that query_text is just a query
    query_words = doc_preprocess(query_text,0)  # Preprocess query
     
    n_commun = np.zeros([len(keydataset),1])
    
    for i in range(len(keydataset)):   # For every document in the dataset
        for query_word in query_words: # For every word of the query
                if (query_word in keydataset[i]):
                    n_commun[i] += 1
    
# #n_commun now has the number of words in commun among the query and the keywords
# #Now we rank according to this. The more words, the better
#indexes = np.array(range(0,len(keydataset)))
#together = zip(n_commun, indexes)  # Zip both arrays together
#sorted_together =  sorted(together, reverse=True) # Sorted Decreasing order
#ordered_indexes = [x[1] for x in sorted_together]
# We dischard the previous since there are a lot of documents that share the same
# number of keywords and making a ranking with them is unfair
    return n_commun
    
    







