# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 12:47:13 2014
Modified Jul 9, 2015

@author: jarenas
"""

import json
import numpy as np
import shutil
import time
import subprocess
import os
from sklearn.preprocessing import normalize


def is_integer(user_str):
    try:
        return int(user_str)
    except ValueError:
        return None


def is_float(user_str):
    try:
        return float(user_str)
    except ValueError:
        return None
        
        
def var_num_keyboard(vartype,default,question):
    """Lee una variable numerica del teclado
    Parametros de entrada:
        * vartype: Tipo de variable de entrada: 'int' or 'float'
        * default: Valor por defecto que se asignal a la variable
        * question: Pregunta al usuario
    """
    aux = raw_input(question + ' [' + str(default) + ']: ')
    if vartype == 'int':
        aux2 = is_integer(aux)
    else:
        aux2 = is_float(aux)
    if aux2 > 0:
        return aux2
    else:
        if aux != '':
            print 'El valor proporcionado no es valido'
        return default


def lee_vocabulario(vocab_filename):
    """Lee el vocabulario del modelo que se encuentra en el fichero indicado
    Devuelve dos diccionarios, uno usando las palabras como llave, y el otro
    utilizando el id de la palabra como clave
    
    Parametro de entrada:
        * vocab_filename     : string con la ruta al vocabulario

    Salida: (vocab_w2id,vocab_id2w)
        * vocab_w2id         : Diccionario {pal_i : id_pal_i}
        * vocab_id2w         : Diccionario {i     : pal_i}
    """
    vocab_w2id = {}
    vocab_id2w = {}
    with open(vocab_filename,'rb') as f:
        for i,line in enumerate(f):
            vocab_w2id[line.strip()] = i
            vocab_id2w[str(i)] =  line.strip()

    return (vocab_w2id,vocab_id2w)


class TopicModel(object):
    
    def __init__(self, betas, thetas, alphas):
        self.betas = betas
        self.thetas = thetas
        self.alphas = alphas
        self.ntopics = self.thetas.shape[1]
        self.size_vocab = self.betas.shape[1]
        #Calculamos betas con down-scoring
        self.betas_ds = np.copy(betas)
        if np.min(self.betas_ds) < 1e-12:
            self.betas_ds += 1e-12
        deno = np.reshape((sum(np.log(self.betas_ds))/self.ntopics),(self.size_vocab,1))
        deno = np.ones( (self.ntopics,1) ).dot(deno.T)
        self.betas_ds = self.betas_ds * (np.log(self.betas_ds) - deno)

        if np.min(self.thetas) < 1e-12:
            self.thetas += 1e-12
        if np.min(self.betas) < 1e-12:
            self.betas += 1e-12


    def muestra_perfiles(self,vocab_filename,n_palabras,tfidf):
        """Muestra los perfiles del modelo lda
        Parametros de entrada:
            * vocab_filename     : string con la ruta al vocabulario
            * n_palabas          : Número de palabras a mostrar para cada perfil
            * tfidf              : Si True, se hace downscaling de palabras poco
                                   específicas (Blei and Lafferty, 2009)
        """
        (vocab_w2id,vocab_id2w) = lee_vocabulario(vocab_filename)
        descripciones = []
        for i in range(self.ntopics):
            if tfidf:
                words = [vocab_id2w[str(idx2)].replace('XXX','ñ')
                    for idx2 in np.argsort(self.betas_ds[i])[::-1][0:n_palabras]]
            else:
                words = [vocab_id2w[str(idx2)].replace('XXX','ñ')
                    for idx2 in np.argsort(self.betas[i])[::-1][0:n_palabras]]
            descripciones.append(' '.join(words))
            print str(i)+'\t'+str(self.alphas[i]) + '\t' + ' '.join(words)

 
print 'What do you want to do?'
print '\t1 - Visualize the extracted topics'
print '\t2 - Generate file for graphic visualiation'
selection = 0
while (selection not in [1, 2]):
    selection = var_num_keyboard('int',1,'Enter your selection:')

n_words = 0
while (n_words not in range(2,25)):
    n_words = var_num_keyboard('int',10,'How many words per topic?')

vocabfile = ''
while not os.path.isfile(vocabfile):
    vocabfile = raw_input('Enter the location of the vocabulary file: ')

model_path = ''
while not os.path.isfile(model_path + '/final.beta'):
    model_path = raw_input('Path to the folder that contains the results of the LDA: ')


with open(model_path+'/final.other','rb') as f:
    alp = float(f.read().split()[-1])
gammas = np.loadtxt(model_path+'/final.gamma') - alp
gammas[gammas<0] = 0
thetas = normalize(gammas,axis=1,norm='l1')
betas = np.exp(np.loadtxt(model_path+'/final.beta'))
alphas = np.mean(thetas,axis=0)
idx = np.argsort(alphas)[::-1]
alphas = alphas[idx]
betas = betas[idx,:]
thetas = thetas[:,idx]
tmodel = TopicModel(betas,thetas,alphas)


if selection == 1:
    tmodel.muestra_perfiles(vocabfile,n_words,False)
elif selection == 2:
    vocab = file(vocabfile, 'rb').readlines()
    vocab = map(lambda x: x.strip(), vocab)  
        
    lista_topics = []
    for idx in range(tmodel.ntopics):
        lista_children = []
        for idx_words in np.argsort(tmodel.betas[idx,])[::-1][:n_words]:
            word = {'name'      : vocab[idx_words],
                    'size'      : tmodel.betas[idx,idx_words]}
            lista_children.append(word)
        topic={'name'       : '%.2f %%' %(100*tmodel.alphas[idx]),
                'size'       : tmodel.alphas[idx],
                'children'   : lista_children}
        lista_topics.append(topic)
    modelo_dict = {'name'          : 'Modelo de topicos',
                   'children'      : lista_topics}
    with open('flare.json', 'wb') as f:
        json.dump(modelo_dict, f)

