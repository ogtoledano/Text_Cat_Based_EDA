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
    categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','comp.windows.x','sci.electronics']
    newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True,categories=categories)
    newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True,categories=categories)
    texts = newsgroups_train['data']
    labels = newsgroups_train['target']

    log_exp_run = make_logger()
    log_exp_run.experiments("Categories-labels: ")
    log_exp_run.experiments(list(newsgroups_train.target_names))
    log_exp_run.experiments("Dictionary scheme: ")
    log_exp_run.experiments(list(newsgroups_train.keys()))
    log_exp_run.experiments("Number of instances for training: ")
    log_exp_run.experiments(len(newsgroups_train['data']))
    log_exp_run.experiments("Number of instances for testing: ")
    log_exp_run.experiments(len(newsgroups_test['data']))

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
    if not os.path.exists(wdir + '/datasets/dataset_train_20ng_nosw_six_labels'):
        dataset_train['features']=pad_sequences(sequences_train, maxlen=max_sequence_length)#[0:5]
        dataset_train['labels']=labels#[0:5]
        torch.save(dataset_train, wdir + "/datasets/dataset_train_20ng_nosw_six_labels")

    dataset_test = {'features': [], 'labels': []}
    texts = newsgroups_test['data']
    labels = newsgroups_test['target']
    removing_stop_words(texts)
    sequences_test = tokenizer.texts_to_sequences(texts)

    if not os.path.exists(wdir + '/datasets/dataset_test_20ng_nosw_six_labels'):
        dataset_test['features']=pad_sequences(sequences_test, maxlen=max_sequence_length)#[0:5]
        dataset_test['labels']=labels#[0:5]
        torch.save(dataset_test, wdir + "/datasets/dataset_test_20ng_nosw_six_labels")

    if not os.path.exists(wdir + '/datasets/dictionary_20ng_nosw_six_labels'):
        torch.save(word_index, wdir + "/datasets/dictionary_20ng_nosw_six_labels")


if __name__ == "__main__":
    build_dataset_and_dict()
