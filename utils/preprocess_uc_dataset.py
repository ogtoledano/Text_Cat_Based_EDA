import pandas as pd
import os
import torch
import nltk

import pandas as pd
from nltk.corpus import stopwords

from utils.logging_custom import make_logger


from sklearn.model_selection import train_test_split

MAX_LEN_SENTENCE=0
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def removing_stop_words(texts):
    # Removing stop words
    stop_words = set(stopwords.words('spanish'))
    for i, text in enumerate(texts):
        tokens = nltk.word_tokenize(text)
        sentence = [word for word in tokens if word not in stop_words]
        texts[i] = ' '.join(sentence)


def build_dataset_and_dict():
    os.chdir('../')
    wdir=os.getcwd()
    file_train=wdir+"/datasets/dataset.xlsx"
    df=pd.read_excel(r''+file_train, engine='openpyxl')
    print(df.iloc[0,1])
    print(df.iloc[350,2])
    print(df.shape)

    y = []
    X = []
    labels = ['1', '2', '3', '4', '5']

    for i in range(df.shape[0]):
        whole_text = str(df.iloc[i, 0]) + ": " + str(df.iloc[i, 1])
        X.append(whole_text)
        y.append(int(df.iloc[i, 2]-1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=142)
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
    tokenizer = Tokenizer(num_words=max_nb_words)
    tokenizer.fit_on_texts(texts)
    sequences_train = tokenizer.texts_to_sequences(texts)

    wdir = os.getcwd()
    if not os.path.exists(wdir + '/datasets/dataset_train_uc_sentiment_nosw'):
        dataset_train['features'] = pad_sequences(sequences_train, maxlen=max_sequence_length)  # [0:5]
        dataset_train['labels'] = labels_target  # [0:5]
        torch.save(dataset_train, wdir + "/datasets/dataset_train_uc_sentiment_nosw")

    dataset_test = {'features': [], 'labels': []}
    texts = bbc_test['data']
    labels_target = bbc_test['target']
    removing_stop_words(texts)

    sequences_test = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    log_exp_run.experiments("Found unique tokens: " + str(len(word_index)))

    if not os.path.exists(wdir + '/datasets/dataset_test_uc_sentiment_nosw'):
        dataset_test['features'] = pad_sequences(sequences_test, maxlen=max_sequence_length)  # [0:5]
        dataset_test['labels'] = labels_target  # [0:5]
        torch.save(dataset_test, wdir + "/datasets/dataset_test_uc_sentiment_nosw")

    if not os.path.exists(wdir + '/datasets/dictionary_uc_sentiment_nosw'):
        torch.save(word_index, wdir + "/datasets/dictionary_uc_sentiment_nosw")


if __name__ == "__main__":
    build_dataset_and_dict()
