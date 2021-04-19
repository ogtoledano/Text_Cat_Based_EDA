import sys

sys.path.append('..\\..\\Text_Cat_Based_EDA')
sys.path.append('..\\..\\Text_Cat_Based_EDA\\utils')
sys.path.append('..\\..\\Text_Cat_Based_EDA\\scripts')
sys.path.append('..\\..\\Text_Cat_Based_EDA\\pretrained_models')
sys.path.append('..\\..\\Text_Cat_Based_EDA\\evolutionary_algorithms')

from evolutionary_algorithms.model_cnn_builder import ModelCNN

import torch
from evolutionary_algorithms.trainer import Trainer
import time

# ------ Scikit-learn ----------------------------------------------------------+
from sklearn.model_selection import GridSearchCV
from skorch.callbacks import Checkpoint, LoadInitState,EarlyStopping


def train_model_sgd(dic_param, log_exp_run, wdir, device, tensor_embedding, train_data, test_data,gscv_best_model):
    # Defining a param distribution for hyperparameter-tuning for model and fit params
    param_grid = {
        'lr': dic_param['alpha_distribution'],
        'optimizer__momentum': dic_param['momentum_distribution'],
        'mode': ["SGD"]  # Modes: Adam,SGD
    }

    fit_param = {
        'patientia': dic_param['sgd_early_stopping_patientia'],
        'min_diference': dic_param['sgd_min_difference'],
        'checkpoint_path': wdir + "checkpoints/"
    }

    checkpoint = Checkpoint(dirname=fit_param['checkpoint_path'], f_params=dic_param['f_params_name'],
                            f_optimizer=dic_param['f_optimizer_name'], f_history=dic_param['f_history_name'],
                            f_criterion=dic_param['f_criterion_name'],
                            monitor=None)

    load_state = LoadInitState(checkpoint)

    # Defining skorch-based neural network
    model = Trainer(
        module=ModelCNN,
        module__word_embedding_size=dic_param['word_embedding_size'],
        module__labels=dic_param['labels'],
        module__weights_tensor=tensor_embedding,
        module__batch_size=dic_param['sgd_batch_size'],
        max_epochs=dic_param['epochs_gs_cv'],
        iterator_train__shuffle=True,
        criterion=torch.nn.CrossEntropyLoss,
        train_split=None,
        device=device,
        callbacks=[checkpoint],
        optimizer=torch.optim.SGD,
        mode="SGD"
        # optimizer__weight_decay=dic_param['l2_reg'] #L2 regularization
    )

    # Defining GridSearch using k-fold cross validation
    log_exp_run.experiments("GridSearch using k-fold cross validation with for SGD")
    start_time = time.time()
    gs = GridSearchCV(model, param_grid, cv=dic_param['grid_search_cross_val_cv'], verbose=2)

    if gscv_best_model is None:
        gs.fit(train_data, fit_param=fit_param)
        log_exp_run.experiments(
            "Time elapsed for GridSearch using k-fold cross validation with k=5 for SGD: " + str(
                time.time() - start_time))

        log_exp_run.experiments("Best param estimated for SGD: ")
        log_exp_run.experiments(gs.best_params_)
        log_exp_run.experiments("Best score for SGD: ")
        log_exp_run.experiments(gs.best_score_)
        log_exp_run.experiments("GridSearch scores")
        log_exp_run.experiments(gs.cv_results_)
        gscv_best_model = gs.best_estimator_

    best_model = gscv_best_model

    best_model.set_params(max_epochs=dic_param['epochs'])
    start_time = time.time()
    best_model.fit(train_data, fit_param=fit_param)
    log_exp_run.experiments("Time elapsed for SGD : " + str(time.time() - start_time))
    best_model.score(test_data)
    best_model.score(train_data)
    log_exp_run.experiments("SGD as optimizer: Process ends successfully!")
    log_exp_run.experiments("--------------------------\n\n\n")
    return gscv_best_model


def train_model_adam(dic_param, log_exp_run, wdir, device, tensor_embedding, train_data, test_data,gscv_best_model):
    # Defining a param distribution for hyperparameter-tuning for model and fit params
    param_grid = {
        'lr': dic_param['alpha_distribution'],
        'mode': ["Adam"]  # Modes: Adam,SGD
    }

    fit_param = {
        'patientia': dic_param['sgd_early_stopping_patientia'],
        'min_diference': dic_param['sgd_min_difference'],
        'checkpoint_path': wdir + "checkpoints/"
    }

    checkpoint = Checkpoint(dirname=fit_param['checkpoint_path'], f_params=dic_param['f_params_name'],
                            f_optimizer=dic_param['f_optimizer_name'], f_history=dic_param['f_history_name'],
                            f_criterion=dic_param['f_criterion_name'],
                            monitor=None)

    load_state = LoadInitState(checkpoint)

    # Defining skorch-based neural network
    model = Trainer(
        module=ModelCNN,
        module__word_embedding_size=dic_param['word_embedding_size'],
        module__labels=dic_param['labels'],
        module__weights_tensor=tensor_embedding,
        module__batch_size=dic_param['sgd_batch_size'],
        max_epochs=dic_param['epochs_gs_cv'],
        iterator_train__shuffle=True,
        criterion=torch.nn.CrossEntropyLoss,
        train_split=None,
        device=device,
        callbacks=[checkpoint],
        optimizer=torch.optim.Adam,
        mode="Adam"
        # optimizer__weight_decay=dic_param['l2_reg'] #L2 regularization
    )

    # model.initialize()
    # print(summary(model.module_,torch.zeros((1,1000),dtype=torch.long), show_input=True))

    # Defining GridSearch using k-fold cross validation
    log_exp_run.experiments("GridSearch using k-fold cross validation with for Adam")
    start_time = time.time()
    gs = GridSearchCV(model, param_grid, cv=dic_param['grid_search_cross_val_cv'], verbose=2)

    if gscv_best_model is None:
        gs.fit(train_data, fit_param=fit_param)

        log_exp_run.experiments(
            "Time elapsed for GridSearch using k-fold cross validation with k=5 for Adam: " + str(
                time.time() - start_time))

        log_exp_run.experiments("Best param estimated for Adam: ")
        log_exp_run.experiments(gs.best_params_)
        log_exp_run.experiments("Best score for Adam: ")
        log_exp_run.experiments(gs.best_score_)
        log_exp_run.experiments("GridSearch scores")
        log_exp_run.experiments(gs.cv_results_)
        gscv_best_model = gs.best_estimator_

    best_model = gscv_best_model

    best_model.set_params(max_epochs=dic_param['epochs'])
    start_time = time.time()
    best_model.fit(train_data, fit_param=fit_param)
    log_exp_run.experiments("Time elapsed for Adam : " + str(time.time() - start_time))
    best_model.score(test_data)
    best_model.score(train_data)
    log_exp_run.experiments("Adam as optimizer: Process ends successfully!")
    log_exp_run.experiments("--------------------------\n\n\n")
    return gscv_best_model

