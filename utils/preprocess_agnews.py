# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------+
#
# @author: Doctorini
# This module preprocces a text file dataset: 20newsgroups removing first
# stop_words and discretize each word form dictionary
#------------------------------------------------------------------------------+

import sys
import os
import torch
import nltk
import sys



sys.path.append('..\\..\\Text_Cat_Based_EDA')
sys.path.append('..\\..\\Text_Cat_Based_EDA\\utils')
from utils.embedding_builder import build_word_embedding,build_tensor
from nltk.corpus import stopwords
from utils.custom_dataloader import VectorsDataloader
from torch.utils.data import DataLoader
from utils.logging_custom import make_logger
from sklearn.datasets import fetch_20newsgroups,fetch_20newsgroups_vectorized
import pandas as pd
from sklearn.model_selection import train_test_split
from random import sample

MAX_LEN_SENTENCE=0

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def removing_stop_words(texts):
    # Removing stop words
    stop_words = set(stopwords.words('english'))
    for i, text in enumerate(texts):
        tokens = nltk.word_tokenize(text)
        sentence = [word for word in tokens if word not in stop_words]
        texts[i] = ' '.join(sentence)

def build_dataset_and_dict():
    path_dataset="C:\\Users\\StarWar\\Desktop\\AGnews"
    X_train = []
    y_train = []

    X_test=[]
    y_test=[]
    labels=['World','Sports','Business','Sci/Tech']

    file_train=path_dataset+"/train.txt"
    file_test = path_dataset + "/test.txt"

    f = open(file_train, "r")
    lines=f.readlines()
    for line in lines:
        text=line.split(',')
        X_train.append(text[1]+" "+text[2])
        y_train.append(text[0][1])

    f = open(file_test, "r")
    lines = f.readlines()
    for line in lines:
        text = line.split(',')
        X_test.append(text[1] + " " + text[2])
        y_test.append(text[0][1])

    bbc_train={'data':X_train,'target':y_train}
    bbc_test = {'data': X_test, 'target': y_test}

    texts = bbc_train['data']
    labels_target = bbc_train['target']

    print(texts[0])

    log_exp_run = make_logger()
    log_exp_run.experiments("Categories-labels: ")
    log_exp_run.experiments(labels)
    log_exp_run.experiments("Number of instances for training: ")
    log_exp_run.experiments(len(bbc_train['data']))
    log_exp_run.experiments("Number of instances for testing: ")
    log_exp_run.experiments(len(bbc_test['data']))

    removing_stop_words(texts)
    dataset_train = {'features': [], 'labels': []}
    max_sequence_length = 1000
    max_nb_words = 2000
    tokenizer=Tokenizer(num_words=max_nb_words)
    tokenizer.fit_on_texts(texts)
    sequences_train=tokenizer.texts_to_sequences(texts)
    word_index=tokenizer.word_index
    log_exp_run.experiments("Found unique tokens: "+str(len(word_index)))

    wdir = os.getcwd()
    if not os.path.exists(wdir + '/datasets/dataset_train_ag_news_nosw'):
        dataset_train['features']=pad_sequences(sequences_train, maxlen=max_sequence_length)#[0:5]
        dataset_train['labels']=labels_target#[0:5]
        torch.save(dataset_train, wdir + "/datasets/dataset_train_ag_news_nosw")

    dataset_test = {'features': [], 'labels': []}
    texts = bbc_test['data']
    labels_target = bbc_test['target']
    removing_stop_words(texts)
    sequences_test = tokenizer.texts_to_sequences(texts)

    if not os.path.exists(wdir + '/datasets/dataset_test_ag_news_nosw'):
        dataset_test['features']=pad_sequences(sequences_test, maxlen=max_sequence_length)#[0:5]
        dataset_test['labels']=labels_target#[0:5]
        torch.save(dataset_test, wdir + "/datasets/dataset_test_ag_news_nosw")

    if not os.path.exists(wdir + '/datasets/dictionary_ag_news_nosw'):
        torch.save(word_index, wdir + "/datasets/dictionary_ag_news_nosw")


if __name__ == "__main__":
    build_dataset_and_dict()
