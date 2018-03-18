

#==============================================================================
#  READ AND STRUCTA THE TEXT OF THE HTML DOCUMENTS FROM THE WEBSITE FOLDER
#==============================================================================


import time  # Just for debuggind commented parts
import os
from bs4 import BeautifulSoup  # For HTML web treatment.
import utilfunc as uf
import pickle   # For exporting and inmportig the lists into files
from urllib import urlopen     # For HTML web download-search

def preprocess_web_corpus (web_folder_path):
#This code:
#    Reads all the HTML documents downloaded from a website.
#    Process the HTML with beatifulSoup to get the body and other elements
#    Preprocess the document up to Steaming 
#    Obtains info from the structure and title and shit
#    Returns tuples ( Corpus_preprocessed, Metadata)
    #==============================================================================
    # Read all the HTML files in the folder and subfolders and preprocess the HTMLs.
    #==============================================================================
 

    Corpus_prep = [];
    # List with all the corpus documents. Every element of the list is an 
    # array with the preprocessed words(tokens)  (tokenization, steaming...)
    
    Metadata_v = [[],[]]  # It has 2 parameters:
#                                - Resource Name: To be able to open the URL afterwards
#                                - Key-words: Special words that have more importance.
#                                    - There words may be: The title of the recepi, the folders containing it....
#                                    
    # os.walk will help us walking through the directories.
    for dirName, subdirList, fileList in os.walk(web_folder_path):
        # For every folder, going top - down
#        print('Found directory: %s' % dirName)
        
        # For every file in the document (HTML file will be)
        for fname in fileList:
            
            # Read the file
            path = dirName + '/' + fname;
            fd = open(path, 'r') 
            doc_HTML = fd.read()
            fd.close()
            
            # Use BeutifulSoup to process the HTML, get tittle, plain text...
            soup = BeautifulSoup(doc_HTML)  # Transform plain text HTML into soup structure
           
#==============================================================================
#             # Get all the plain text and preprocess it:
#==============================================================================
           
            # First eliminate all the JS code and CSS
            for elem in soup.findAll(['script', 'style']):
                elem.extract()
            
            # DELETE SPECIFIC PARTS OF THE WEB THAT CONTAIN WORDS FROM OTHER RECIPES
            # THAT WILL MAKE THE SYSTEM FAIL
            if (web_folder_path == './www.homecookingadventure.com'):
                 
                 rem = soup.findAll("a")  # Links (To anything)
                 if (rem != None):
                     for elem in rem:
                         elem.extract()
                 rem = soup.find("div", {"id": "side_right"})   # menu, latest_links-..
                 if (rem != None):
                     rem.extract()
                 rem = soup.find("div", {"id": "posted_comments"}) # Comments
                 if (rem != None):
                     rem.extract()
                 rem = soup.find("div", {"id": "right_col"})  # You may also like
                 if (rem != None):
                     rem.extract() 
                 text = soup.get_text()
                 
                # Get Special elements from HTML
                 title = str(soup.title.string)
                 keywords = title.split("::")[0]  # to eliminate the endind ":: Homeadventure blabla"
                 print keywords
                 keywords = uf.doc_preprocess(keywords,0)  # We preprocess them
                
            if (web_folder_path == './allrecipes.com'):
                 useful_text = " "
                 keywords = " "
                 useful = soup.find("div", {"class": "directLeft"})  # Cooking
                 useful_text += " " + useful.text
                 useful = soup.find("div", {"class": "ingred-left"})   # Ingredientes
                 useful_text += " " + useful.text
                 useful = soup.find("span", {"id": "lblDescription"})   # Description
                 useful_text += " " + useful.text
                 
                 useful = soup.findAll("div", {"itemtype": "http://data-vocabulary.org/Breadcrumb"})  # Directori (Dessert > Bannaan )
                 for elem in useful:
                     useful_text += " " + elem.text
                     keywords += " " + elem.text
                     
                 useful = soup.find("h1", {"id": "itemTitle"}) # Title
                 useful_text += " " + useful.text
                 keywords += " " + useful.text
                 print useful.text
#                  <meta itemprop="ratingValue" content="2.142857"> 
                 text = useful_text  # Gets all the text from the HTLM, eliminating the marks
                 keywords = uf.doc_preprocess(keywords,0)  # We preprocess them
                 keywords = keywords[2:]   # To eliminate tags home and recipes
#            print text
#            time.sleep(10)
            
            preprocessed_text = uf.doc_preprocess(text,0) # Preprocess to get the tokens
            Corpus_prep.append(preprocessed_text)
            
            URL  = "http://" + path[2:]
            Metadata_v[0].append(URL)
            Metadata_v[1].append(keywords)
    return (Corpus_prep, Metadata_v)
    

def get_corpus_TFIDF_vectors (Corpus_prep):
    #==============================================================================
    # Get the TFIDF Vectors of the preprocessed data
    #==============================================================================
    Corpus_v = []
    N_doc = len(Corpus_prep)
    for i in range(N_doc):
        words_tdidf = uf.tf_idf_doc(Corpus_prep[i], Corpus_prep)
        words_tdidf = uf.doc_sort_words(words_tdidf) # Optional, just order the vector semantically
        Corpus_v.append(words_tdidf)
        # For every document, the array docs_Vector has a list of two elements (words, tf-idfs)
    # List with all the corpus documents in TF-IDF vector form. 
    #Every element (document) of the list has two arrays:
    #    - words: Array of the relevant words the document has
    #    - tfidfs: Array with the tfidf values of those words
    return Corpus_v
    
# USE pickle library to export it into a file, and read it afterwards
def write_corpus_file (Corpus_v,Metadata_v, corpus_file):
    with open(corpus_file, 'wb') as f:
        pickle.dump((Corpus_v, Metadata_v), f)
        
def load_corpus_file(corpus_file):
    with open(corpus_file, 'rb') as f:
        corpus_v = pickle.load(f)
    return corpus_v
    

def get_text_web_query (web_url):
    # This function downloads the recipe given by the url and tries to find a similar one.
    html = urlopen(web_url).read() 
    soup = BeautifulSoup(html)  # Transform plain text HTML into soup structure
    # First eliminate all the JS code and CSS and hyperlinks (they usually have the names of other recipes)
    for elem in soup.findAll(['script', 'style', 'a' ]):
        elem.extract()
    
    
    text = soup.get_text()  # Gets all the text from the HTLM, eliminating the marks
    return text
    
#==============================================================================
# import glob
# for infile in glob.glob( os.path.join(website_folder, '*') ):
#     print "current file is: " + infile
#==============================================================================


# os.walk knowloedge !!!


# dirName, subdirList, fileList in os.walk(website_folder):
#dirName has the name of the folder that os.walk is processing.
#subdirList has the list of subfolders
#fileList has the list of files in dirName.
#
#We perform a recursive search with the for, going through all files
#dirName, subdirList, fileList in os.walk(website_folder)
#print dirName
#print subdirList
#print fileList









#==============================================================================
#                 rem = soup.find("div", {"id": "latest_articles"})
#                 if (rem != None):
#                     rem.extract()
#                 rem = soup.find("div", {"id": "latest_recipes"})
#                 if (rem != None):
#                     rem.extract()
#                 rem = soup.find("div", {"id": "posted_comments"})
#                 if (rem != None):
#                     rem.extract()
#                 soup.findAll("div", { "class" : "menu_sec" })
#                 if (rem != None):
#                     print rem
#                     for elem in rem:
#                         elem.extract()
#==============================================================================
      