
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------+
#
# @author: Doctorini
# This module preprocces a text file dataset: Ecured five tags removing first
# stop_words and discretize each word form dictionary
#------------------------------------------------------------------------------+

import os
import sys
from html.parser import HTMLParser

import nltk
import torch

sys.path.append('..\\..\\Text_Cat_Based_EDA')
sys.path.append('..\\..\\Text_Cat_Based_EDA\\utils')
from nltk.corpus import stopwords
from utils.logging_custom import make_logger
from sklearn.model_selection import train_test_split

MAX_LEN_SENTENCE=0

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class MyHTMLParser(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)
        self.isDocument = False
        self.paragraph = ""
        self.data=[]
        self.isParagraph=False

    def handle_starttag(self, tag, attrs):
        if tag == 'doc':
            self.isDocument = True

        if tag == 'p':
            self.isParagraph = True

    def handle_endtag(self, tag):
        if tag == 'doc':
            self.isDocument = False
            self.data.append(self.paragraph)
            self.paragraph = ""

        if tag == 'p':
            self.isParagraph = False

    def handle_data(self, data):
        if self.isDocument:
            if self.isParagraph:
                self.paragraph += data


def removing_stop_words(texts):
    # Removing stop words
    stop_words = set(stopwords.words('spanish'))
    for i, text in enumerate(texts):
        tokens = nltk.word_tokenize(text)
        sentence = [word for word in tokens if word not in stop_words]
        texts[i] = ' '.join(sentence)


def stop_words_count_and_length(texts):
    # Counting stop words
    stop_words = set(stopwords.words('spanish'))
    count_sw=0
    total_length=0
    for i, text in enumerate(texts):
        tokens = nltk.word_tokenize(text)
        total_length+=len(tokens)
        for word in tokens:
            if word in stop_words:
                count_sw += 1
    return total_length,count_sw

def parse_documents_from_html_format(text_html):
    parser=MyHTMLParser()
    parser.feed(text_html)
    parser.close()
    return parser


def build_dataset_and_dict():
    path_dataset="C:\\Users\\Laptop\\Desktop\\ecured_five_tags"

    labels = ['ciencia', 'cultura', 'deporte', 'historia', 'salud']

    y = []
    X = []

    for i, label in enumerate(labels):
        folder = path_dataset + "/" + label
        for file in os.listdir(folder):
            total_text=[]
            f = open(folder + "/" + file, "r",encoding='UTF-8')
            text = f.read()
            parse=parse_documents_from_html_format(text)
            for pattern in parse.data:
                X.append(pattern)
                total_text.append(pattern)
                y.append(i)
        total_length, count_sw=stop_words_count_and_length(total_text)
        print("Found tokens: {} for label: {}, and count stop-words {}".format(total_length, label,count_sw))
        total_text=[]

    X_train, X_test, y_train,y_test=train_test_split(X, y, test_size=0.30, random_state=142)
    bbc_train={'data':X_train,'target':y_train}
    bbc_test = {'data': X_test, 'target': y_test}
    texts = bbc_train['data']
    labels_target = bbc_train['target']
    os.chdir("../")
    log_exp_run = make_logger()

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
    if not os.path.exists(wdir + '/datasets/dataset_train_ecured_nosw'):
        dataset_train['features']=pad_sequences(sequences_train, maxlen=max_sequence_length)#[0:5]
        dataset_train['labels']=labels_target#[0:5]
        torch.save(dataset_train, wdir + "/datasets/dataset_train_ecured_nosw")

    dataset_test = {'features': [], 'labels': []}
    texts = bbc_test['data']
    labels_target = bbc_test['target']
    removing_stop_words(texts)
    sequences_test = tokenizer.texts_to_sequences(texts)

    if not os.path.exists(wdir + '/datasets/dataset_test_ecured_nosw'):
        dataset_test['features']=pad_sequences(sequences_test, maxlen=max_sequence_length)#[0:5]
        dataset_test['labels']=labels_target#[0:5]
        torch.save(dataset_test, wdir + "/datasets/dataset_test_ecured_nosw")

    if not os.path.exists(wdir + '/datasets/dictionary_ecured_nosw'):
        torch.save(word_index, wdir + "/datasets/dictionary_ecured_nosw")


if __name__ == "__main__":
    build_dataset_and_dict()