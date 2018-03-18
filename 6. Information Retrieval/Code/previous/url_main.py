from bs4 import BeautifulSoup  # For HTML web treatment.
from urllib import urlopen     # For HTML web download-search

#==============================================================================
# url = "http://nltk.org/book/ch01.html"
# html = urlopen(url).read() 
# soup = BeautifulSoup(html)  # Transform plain text HTML into soup structure
# 
# #==============================================================================
# # print soup.title
# # print soup.title.name
# # print soup.title.string
# # print soup.title.parent.name
# # print soup.p
# # print soup.p('class')
# # print soup.a
# #==============================================================================
# 
# html_text = soup.get_text()  # Gets all the text from the HTLM, eliminating the marks
#==============================================================================


from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
# Functions for tokenizing

s = 'Good muffins cost $3.88\nin New York. Please buy me two of them. Thanks'
s_sent = sent_tokenize(s)           # Tokenizes (separates) sentences.
s_word  =  word_tokenize(s)         # Tokenizes (separates) words.

#print s_sent
#print s_word


s_word[4].isalpha()                 # Checks if the word is alphanumeric

#==============================================================================
# # PYTHON FORM OF GETTING FRECUENCIES WITHOUT NLTK
# from collections import Counter    # For counting the words
#                                    # Uses a counter object (dictionary subclass)
# frec = Counter(s_word)
# frec = frec.most_common()       # Returns all items in a tuple list way
# print frec
# 
# frec = sorted(frec,None,lambda x: x[1],True)  # To order them in decreasing order we set reverse = true
# print frec
#==============================================================================

# Getting frequencies with NLTK

from nltk import FreqDist
fdist = FreqDist(s_word)

frec = fdist.most_common()
print frec
print fdist['.']
word_freq = fdist.items()

