# SiGEDA: Text Categorization Based on EDA algorithms

This is a project that allows text categorization for English and Spanish documents. It is based on a deep neural architecture using Convolutional Neural Networks, and word embedding (GloVE) as word vector representation. The neural network is trained by applying gradient-based methods (SGD and Adam), and combines the training with a new hybrid learning strategy via Estimation Distribution Algorithms (EDA). Includes three evolutionary strategies such Univariate Marginal Distribution Algorithm for countinuos domain (UMDAc), Estimation of Multivariate Normal Algorithm (EMNA) and Covariance Matrix Adaptation - Evolutionary Strategy (CMA-ES).

*********** update at April 18, 2021 *************

## Requirements

In order to run this project you will need a working installation of:

+ deap
+ gensim
+ scikit-learn
+ pandas
+ numpy
+ nltk

For loading pre-trained models, Dataloaders, hyper-parameters tuning, CNN training and testing, you will need:
+ torch == 1.2.0+cu92
+ skorch == 0.9.0

## Pre-trained models

We use two word embedding (English, Spanish) pre-trained GloVe (Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014):

+ English GloVe with 6 Billions words and 100 dimension vector. Pre-trained model is available at [here](http://nlp.stanford.edu/data/glove.6B.zip)
+ Spanish Glove with 1.4 Billions words and 300 dimension vector. Pre-trained model is available at [here](http://dcc.uchile.cl/~jperez/word-embeddings/glove-sbwc.i25.vec.gz)

Download pre-trained models and put them into `utils/pretrained_models` folder 

## Datasets

+ BBC Sport (Derek Greene and Pádraig Cunningham, 2006). Data is available at [here](http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip)
+ BBC News (Derek Greene and Pádraig Cunningham, 2006). Data is available at [here](http://mlg.ucd.ie/files/datasets/bbcsport-fulltext.zip)
+ Youtube+Spam+Collection (Tulio C. Alberto et. al, 2016). Data is available at [here](https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection)

## scripts/arguments.txt file

This file contains the arguments and hyper-parameters essential to run the code. 

It is structured as follows:

```shell
argument_key=value
```
The following is the meaning of each argument key:
```shell
dataset_train -> Name of pre-processed dataset for training (pre-proceced dataset in datasets folder)
dataset_test_ -> Name of pre-processed dataset for for testing 
dataset_dictionary -> Name of dictionary for dataset
name_log_experiments_result -> Log file name for algorithms output (at experiments folder)
epochs_gs_cv -> Amount of epochs for Grid-search CV
cuda_device_id -> CUDA device id
generations -> Amount of generations
num_executions -> Amount of experiments execution
alpha_distribution -> Grid for learning rate 
momentum_distribution -> Grid for momentum
centroid_distribution -> Grid for initial centroid
sigma_distribution -> Grid for initial variance
sgd_early_stopping_patientia -> Early stopping criteria patientia
sgd_batch_size -> Batch size
grid_search_cross_val_cv= Amount of folds for Grid-search CV
individuals -> Amount of individuals
cnn_checkpoint -> Checkpoint name for learning learning full-connected layer by EDA optimization, weights of the full-connected layer. 
cnn_optimizer -> Gradient-based method for learning convolutional filters (SGD, Adam) 
sigma -> Initial variance for EDA optimization
centroid -> Initial centroid for EDA optimization
labels -> Amount of labels/categories for classification problem
word_embedding_size -> Embedding dimension
word_embedding_pretrained_glove_language -> Pre-trained embedding language (English, Spanish)
f_params_name -> Checkpoint name for CNN weights (f_param). All checkpoint are saved in checkpoint folder.
f_optimizer_name-> Checkpoint name for CNN optimizer
f_history_name -> JSON file name for history
f_criterion_name -> Checkpoint name for CNN criterion
```
## Project structure
The project has the following structure:
+ `checkpoints` For saving automatically Pytorch Tensors models. Useful for loading kernel parameters in each convolutional layer from previous learning algorithms
+ `datasets ` Location for pre-processed datasets in Pytorch Tensors.
+ `experiments` Logging folder for saving algorithms results.
+ `scripts` Main executables python files.
+ `statistic_test` Scripts for statistic tests.
+ `utils` Pre-processing files and pre-trained models
```bash
├── Text_Cat_Based_EDA
│   ├── checkpoints
│   ├── datasets
│   ├── evolutionary_algorithms
│   │   ├── eda
│   │   │    ├── CUMDA.py
│   │   │    └── EDA.py
│   │   ├── callback_ann.py
│   │   ├── evolutionary_optimizer.py
│   │   ├── model_cnn_builder.py
│   │   └── trainer.py
│   ├── experiments
│   ├── scripts
│   │   ├── arguments.txt
│   │   ├── main_adam.py
│   │   ├── main_eda_cma_es.py
│   │   ├── main_eda_cumda.py
│   │   ├── main_eda_emna.py
│   │   ├── main_gradient_based.py
│   │   └── main_sgd.py
│   ├── statistic_test
│   │   ├── statistic_bbc_news.py
│   │   ├── statistic_bbcsport.py
│   │   ├── statistic_hyer_param_tuning.py
│   │   ├── statistic_hypothesis_test.py
│   │   └── statistic_youtube.py
│   ├── utils
│   │   ├── pretrained_models
│   │   ├── custom_dataloader.py
│   │   ├── embedding_builder.py
│   │   ├── preprocess_bbcnews.py
│   │   ├── preprocess_bbcsports.py
│   │   ├── preprocess_ecured_five_tags.py
│   │   └── preprocess_youtube.py
│   └── run.slurm
├── README.md
└── .gitignore
```

## Run the code

After setting some arguments in the scripts/arguments.txt file, add the project path to the PYTHONPATH.

If you want to train using only Adam optimizer run: 
```shell
python scripts/main_adam.py
```

For training using only SGD optimizer run: 
```shell
python scripts/main_sgd.py
```

**Hybrid method based on EDA optimization:**

Train CNN with specified `cnn_optimizer ` in _scripts/arguments.txt_ using UMDAc for learning full-connected layer:
```shell
python scripts/main_eda_cumda.py
```

Train CNN with specified `cnn_optimizer ` in _scripts/arguments.txt_ using EMNA for learning full-connected layer:
```shell
python scripts/main_eda_emna.py
```

Train CNN with specified `cnn_optimizer ` in _scripts/arguments.txt_ using CMA-ES for learning full-connected layer:
```shell
python scripts/main_eda_cma_es.py
```
