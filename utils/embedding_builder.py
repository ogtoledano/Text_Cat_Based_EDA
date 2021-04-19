# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------+
#
# @author: Doctorini
# This module train or load a word embedding model based on word2vec
# and build a suitable Tensor with padding
#
#------------------------------------------------------------------------------+

import multiprocessing
import os
import sys
import warnings

import gensim
import torch
from nltk.corpus import brown

sys.path.append('..\\..\\Text_Cat_Based_EDA')
sys.path.append('..\\..\\Text_Cat_Based_EDA\\utils')
from utils.logging_custom import make_logger
import numpy as np
import time

# Defining constants
EMBEDDING_SIZE = 100
WINDOW = 5
MIN_COUNT = 5
NEGATIVE_SAMPLING= 15
EPOCHS = 10

warnings.filterwarnings(action='ignore')


# Create or load Word2vec model
def build_word_embedding(url_pretrained_model):
    model=None
    log_exp_run = make_logger()
    if not os.path.exists(url_pretrained_model+"/word2vec"+"_"+str(EMBEDDING_SIZE)+".model"):
        model = gensim.models.Word2Vec(brown.sents(), size=EMBEDDING_SIZE, window=WINDOW, min_count=MIN_COUNT,
                                       negative=NEGATIVE_SAMPLING, iter=EPOCHS, workers=multiprocessing.cpu_count())
        model.save(url_pretrained_model+"/word2vec"+"_"+str(EMBEDDING_SIZE)+".model")
        log_exp_run.experiments("Created and saved word embedding model with:")
        log_exp_run.experiments("EMBEDDING_SIZE: "+ str(EMBEDDING_SIZE))
        log_exp_run.experiments("DICTIONARY LENGTH: " + str(len(model.wv.vocab)))
    else:
        model=gensim.models.Word2Vec.load(url_pretrained_model+"/word2vec"+"_"+str(EMBEDDING_SIZE)+".model")
        log_exp_run.experiments("Loaded word embedding model with:")
        log_exp_run.experiments("EMBEDDING_SIZE: " + str(EMBEDDING_SIZE))
        log_exp_run.experiments("DICTIONARY LENGTH: " + str(len(model.wv.vocab)))

    return model


# Load GloVe from pretrained model and giving dictionary
def build_glove_from_pretrained(url_pretrained_model,url_dictionary):
    embedding_dict={}
    log_exp_run = make_logger()
    file_pretrained = open(url_pretrained_model+"/glove.6B.100d.txt","r",encoding='ANSI')#removing encoding in unix-based os ,encoding='ANSI'
    start_time = time.time()
    lines = file_pretrained.readlines()
    for line in lines:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:],dtype='float32')
        embedding_dict[word] = coefs

    file_pretrained.close()

    log_exp_run.experiments("Loaded word embedding model with GloVe:")
    log_exp_run.experiments("EMBEDDING_SIZE: " + str(len(embedding_dict["the"])))
    log_exp_run.experiments("DICTIONARY LENGTH: " + str(len(embedding_dict)))
    log_exp_run.experiments("Time elapsed for loading embedding vectors from file: " + str(time.time() - start_time))

    word_index=torch.load(url_dictionary)
    embedding_matrix=np.random.random((len(word_index)+1,100))

    log_exp_run.experiments("Length of dictionary of dataset: "+str(len(word_index)))

    for word,i in word_index.items():
        embedding_vector=embedding_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return torch.FloatTensor(embedding_matrix)


def build_spanish_glove_from_pretrained(url_pretrained_model,url_dictionary):
    from gensim.models.keyedvectors import KeyedVectors
    wordvectors_file_vec = url_pretrained_model+'/glove-sbwc.i25.vec'
    cantidad = 100000
    wordvectors = KeyedVectors.load_word2vec_format(wordvectors_file_vec, limit=cantidad)
    embedding_dict = {}
    log_exp_run = make_logger()
    start_time = time.time()
    for word in wordvectors.vocab:
        embedding_dict[word] = np.asarray(wordvectors.wv.get_vector(word), dtype='float32')

    log_exp_run.experiments("Loaded spanish word embedding model with GloVe:")
    log_exp_run.experiments("EMBEDDING_SIZE: " + str(len(embedding_dict["the"])))
    log_exp_run.experiments("DICTIONARY LENGTH: " + str(len(embedding_dict)))
    log_exp_run.experiments("Time elapsed for loading embedding vectors from file: " + str(time.time() - start_time))

    word_index = torch.load(url_dictionary)
    embedding_matrix = np.random.random((len(word_index) + 1, 300))

    log_exp_run.experiments("Length of dictionary of dataset: " + str(len(word_index)))

    for word, i in word_index.items():
        embedding_vector = embedding_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # print(wordvectors.similarity("celular","computadora"))
    # print(wordvectors.most_similar_cosmul(positive=['cantante','grabación'],negative=['concierto']))
    return torch.FloatTensor(embedding_matrix)



def build_tensor(url_path):
    weight = build_word_embedding(url_path).wv.vectors
    tensor_weight = torch.Tensor(weight)
    padding = torch.zeros(EMBEDDING_SIZE).reshape(1, EMBEDDING_SIZE)
    tensor_weight = torch.cat((tensor_weight,padding))
    return tensor_weight


if __name__ == "__main__":
    #os.chdir("../")
    #wdir=os.getcwd()+"/"
    #build_spanish_glove_from_pretrained(wdir+'utils/pretrained_models',wdir+'datasets/dictionary_ecured_nosw')
    pass
    #build_glove_from_pretrained('utils/pretrained_models','datasets/dictionary_20ng')
