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
#print fdist['.']
word_freq = fdist.items()
print word_freq
# Steamming

# With regular expressions

#==============================================================================
# import re
# for i in range(len(s_word)):
#     # Regex notation
#     print re.findall(r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$', s_word[i])
# 
# # With porter Steammer.
# 
# from nltk import PorterStemmer
# raw = 'DENNIS: Listen, strange women lying in ponds distributing swords is no basis for a system of government.'
# tokens = [w.lower() for w in word_tokenize(raw)]
# porter = PorterStemmer()
# print [porter.stem(t) for t in tokens]
# 
# # With porter nltk.LancasterStemmer().
# 
# from nltk import LancasterStemmer
# raw = 'DENNIS: Listen, strange women lying in ponds distributing swords is no basis for a system of government.'
# tokens = [w.lower() for w in word_tokenize(raw)]
# lancaster = LancasterStemmer()
# print [lancaster.stem(t) for t in tokens]
# 
# # Lemmatization
# 
# from nltk import WordNetLemmatizer
# wnl = WordNetLemmatizer()
#==============================================================================
#print [wnl.lemmatize(t) for t in tokens]
