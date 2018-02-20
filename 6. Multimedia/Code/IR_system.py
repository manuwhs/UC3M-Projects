import utilfunc as uf
import corpus_f as cf
import webbrowser  # To open webPage in browser!!
import numpy as np
#==============================================================================
# WEBCRAWLER THEORY.
# First we have to select the crawler that is going to download the website.
# We have modified "pipelines" in order to specify the files we download and
# some properties we extract from them.
# In spider_test.py we have specified some specific webcrawlers to specific
# websites and how they crawl them.
# To execute the Spider:
#montoya@montoya:~$ cd Desktop/SCHEDULE/Multimedia/
#montoya@montoya:~/Desktop/SCHEDULE/Multimedia/scrapy_test$ scrapy crawl dmoz
#==============================================================================

# To eliminate files from recipes: 
#    find -name '*review*.html' | xargs -d \\n rm
#    find -name '*detail*.html' | xargs -d \\n rm
#    find -name '*aspx*.html' | xargs -d \\n rm
#    find -name '*index*.html' | xargs -d \\n rm
#    find . -empty -type d -delete        Delete empty folders
#When we download the webpage, we get some redundant shit files we want to delete.

#==============================================================================
# CORPUS WEBSITES
#==============================================================================
website_folders = [ './www.homecookingadventure.com', "./allrecipes.com"]
# './www.homecookingadventure.com',
#==============================================================================
# FLAGS
#==============================================================================
process_corpus = 0  # Flag that when 1, it reads the webpage folder and proces
                    # all the corpus HTML and gets the features and writes it into a file.
query_type = 0;     # 0 is for plain text as query
                    # 1 is for an URL as query
                    # 2 is for an 

#==============================================================================
# OBTAIN PREPROCESSED CORPUS
#==============================================================================
if (process_corpus == 1):   # If we have to preprocess the web, we do
    # Call this function to preprocess the websites, read the HTML
    Corpus_prep = []
    Metadata_v = [[], []] # URL and keywords
    print "Reading and Preprocessing HTML documents"
    for web_folder in website_folders:
        print "Reading website_folder" + web_folder
        Data_v = cf.preprocess_web_corpus (web_folder) # Preprocess HTML
        Corpus_prep.extend(Data_v[0])
        Metadata_v[0].extend(Data_v[1][0])
        Metadata_v[1].extend(Data_v[1][1])
    print "Obtaining TDIDF vectors"
    Corpus_v = cf.get_corpus_TFIDF_vectors(Corpus_prep) # Get the TFIDF vectors
    
    cf.write_corpus_file (Corpus_v,Metadata_v, "corpus.pkl") # Write file with the corpus all processed
    
else:   # If we dont, we just load the preprocessed file
    Data_v = cf.load_corpus_file("corpus.pkl")
    Corpus_v = Data_v[0]
    Metadata_v = Data_v[1]
#==============================================================================
# GET THE QUERY
#==============================================================================
query_type = 0
query = "eggplant chocolate cucumber roasted !"
#query = "mayonaise macadamia jam salad"
#query = "wine rice salmon"
#query = "egg pepper beef"


# When we have a short text query, usually, the one that wins is that one with
# the best relative frecuency in the HTML, if the word appears in the recepi,
# and the recipe has only few words, then this document will have a lot of similarity.
# Which might not be right, WE CAN GIVE MORE IMPORTANCE TO THE WORDS IN THE TITTLE


if (query_type == 0):  # Plain text query
    text_query = query
elif (query_type == 1): # URL web query
    text_query = cf.get_text_web_query(query)

#==============================================================================
# OBTAIN MOST SIMILAR URLs
#==============================================================================
Simi_v = uf.get_similarities(text_query,Corpus_v)

N_top = 5;

#top_indexes = uf.rank_documents(Simi_v[0], N_top,"euclidean")
#print top_indexes
#top_indexes = uf.rank_documents(Simi_v[1], N_top,"cosine")
#print top_indexes

if (query_type == 0):  # Plain text query
    top_indexes = uf.rank_documents(Simi_v[3], N_top,"tdidf_sum")
elif (query_type == 1): # URL web query
    top_indexes = uf.rank_documents(Simi_v[0], N_top,"euclidean")
#    top_indexes = uf.rank_documents(Simi_v[1], N_top,"cosine")
    
print top_indexes

keywords_corpus = Metadata_v[1]
top_indexes2 = uf.rank_by_keywords(query, keywords_corpus)


#==============================================================================
# Open the top URL documents
#==============================================================================
for i in range(N_top):
    # Open website in a new window if possible
    webbrowser.open(Metadata_v[0][top_indexes[i]] , new = 1)

#query ="http://www.food.com/recipe/blue-moon-burgers-92316"
#query = "http://www.food.com/recipe/dark-chocolate-cake-2496"
#query = "http://www.food.com/recipe/to-die-for-crock-pot-pork-chops-252250"

