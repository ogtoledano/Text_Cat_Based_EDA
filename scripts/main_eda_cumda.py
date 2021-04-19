import sys
sys.path.append('..\\..\\Text_Cat_Based_EDA')
sys.path.append('..\\..\\Text_Cat_Based_EDA\\utils')
sys.path.append('..\\..\\Text_Cat_Based_EDA\\scripts')
sys.path.append('..\\..\\Text_Cat_Based_EDA\\pretrained_models')
sys.path.append('..\\..\\Text_Cat_Based_EDA\\evolutionary_algorithms')

from evolutionary_algorithms.model_cnn_builder import ModelCNN
from utils.embedding_builder import build_glove_from_pretrained, build_spanish_glove_from_pretrained

import torch
from utils.custom_dataloader import CustomDataLoader
from utils.logging_custom import make_logger
from evolutionary_algorithms.evolutionary_optimizer import EDA_Optimizer
from utils.file_arguments_reader import load_param_from_file
import os
import time
from main_gradient_based import train_model_adam, train_model_sgd

if __name__ == "__main__":
    # Load train arguments from file
    os.chdir("../")
    wdir = os.getcwd() + "/" if not os.path.exists("/home/CLUSTER/uclv_ogtoledano/doctorado/Text_Cat_Based_EDA/") else "/home/CLUSTER/uclv_ogtoledano/doctorado/Text_Cat_Based_EDA/"  # only slurm cluster
    dic_param = load_param_from_file(wdir + "scripts/arguments.txt")
    log_exp_run = make_logger(name="" + dic_param['name_log_experiments_result'])
    device = "cuda:" + str(dic_param['cuda_device_id']) if torch.cuda.is_available() else "cpu"

    # Load pre-trained word embedding model with specific language: Spanish or English
    tensor_embedding = build_spanish_glove_from_pretrained(wdir + 'utils/pretrained_models',
                                                            wdir + 'datasets/' + dic_param['dataset_dictionary']) if \
                                                            dic_param['word_embedding_pretrained_glove_language'] == 'Spanish' \
                                                            else build_glove_from_pretrained(wdir + 'utils/pretrained_models',
                                                            wdir + 'datasets/' + dic_param['dataset_dictionary'])

    train_data = CustomDataLoader(wdir + 'datasets/' + dic_param['dataset_train'])
    test_data = CustomDataLoader(wdir + 'datasets/' + dic_param['dataset_test'])

    gscv_best_model = None

    for i in range(dic_param['num_executions']): # Total measurement
        # Do STAGE 1: Learning convolutional layers by gradient-based method
        start_time = time.time()
        if dic_param['cnn_optimizer'] == 'Adam':
            gscv_best_model = train_model_adam(dic_param, log_exp_run, wdir, device, tensor_embedding, train_data, test_data,gscv_best_model)
        if dic_param['cnn_optimizer'] == 'SGD':
            gscv_best_model = train_model_sgd(dic_param, log_exp_run, wdir, device, tensor_embedding, train_data, test_data,gscv_best_model)

        # Defining skorch-based neural network
        model = EDA_Optimizer(
            module=ModelCNN,
            module__word_embedding_size=dic_param['word_embedding_size'],
            module__labels=dic_param['labels'],
            module__weights_tensor=tensor_embedding,
            module__batch_size=dic_param['sgd_batch_size'],
            train_split=None,
            criterion=torch.nn.CrossEntropyLoss,
            device=device,
            sigma=dic_param['sigma'],
            mode="EDA_CUMDA"
        )

        param_distribution = {'sigma': dic_param['sigma_distribution'], 'mode': ["EDA_CUMDA"]}

        param_model = {"generations": dic_param['generations'], 'mode': "EDA_CUMDA",
                       "population_size": dic_param['individuals'],
                       "checkpoint": wdir + "checkpoints/" + dic_param['cnn_checkpoint']}

        # Do STAGE 2: Learning full-connected layer by EDA optimization
        model.fit(train_data, fit_param=param_model)
        log_exp_run.experiments("Time elapsed: " + str(time.time() - start_time))
        model.score(test_data)
        model.score(train_data)
        log_exp_run.experiments(dic_param['cnn_optimizer'] + " + EDA_CUMDA: Process ends successfully!")
        log_exp_run.experiments("--------------------------\n\n\n")
