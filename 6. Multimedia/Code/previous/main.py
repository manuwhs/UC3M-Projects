from nltk.corpus import gutenberg  # Import corpus

import utilfunc as uf
# The u'String' means it is codified in unicode. String.encode("ascii") to convert it. utf-8
#==============================================================================
# for fileid in gutenberg.fileids(): # For every file (fileid is the name of it)
#     num_chars = len(gutenberg.raw(fileid))
#     num_words = len(gutenberg.words(fileid))
#     num_sents = len(gutenberg.sents(fileid))
#     num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
#     print(round(num_chars/num_words), round(num_words/num_sents), round(num_words/num_vocab), fileid)
#==============================================================================

# Get list of documents.
list_of_docs = []
#==============================================================================
# for fileid in gutenberg.fileids():
#     list_of_docs.append(gutenberg.raw(fileid))
#==============================================================================
s1 = 'Pedro, el coche de Jorge se ha caido, va a haber cantus, aun no se sabe fee, se ha pedido party responsible,'
s2 = 'Dani, seria party r. solo para una fiesta, pues va a haber cantus, el sbado. '
s3 = 'CeliaO, party r. la persona que orienta la fiesta, no es un cargo serio, una carlinhada es un evento de motivacion, tenermos training, apredemos que hacemos en best, dirigida a preparar eventos, suele haber 2 carlinhadas, al principio de curso y en febrero.'
s4 = 'Daniv, somo carlinhos porque nuestra mascota es un gato llamado carlinho.'
s5 = 'Pedro, el viernes me gustaria hacer inventario de la office en la tarde, a partir de las3 pm.'
s6 = 'Kaicu, partir de las 5puedo pasasrme. '

list_of_docs.append(s1)
list_of_docs.append(s2)
list_of_docs.append(s3)
list_of_docs.append(s4)
list_of_docs.append(s5)
list_of_docs.append(s6)


# Preprocess texts using stemming.
N_doc = len(list_of_docs)
for i in range(N_doc):
    list_of_docs[i] = uf.doc_preprocess(list_of_docs[i],0) 

# Get the a vector with pairs (words, tdidf)
docs_Vector = []
for i in range(N_doc):
    words_tdidf = uf.tf_idf_doc(list_of_docs[i], list_of_docs)
    words_tdidf = uf.doc_sort_words(words_tdidf) # Optional, just order the vector semantically
    docs_Vector.append(words_tdidf)
    # For every document, the array docs_Vector has a list of two elements (words, tf-idfs)
query = "Pedro, viernes fiesta fefer fee inventario"

simi = uf.get_similarities(query,docs_Vector)
top_indexes = uf.rank_documents(simi, 4)
print top_indexes
