import sys

sys.path.append('..\\..\\Text_Cat_Based_EDA')
sys.path.append('..\\..\\Text_Cat_Based_EDA\\utils')
sys.path.append('..\\..\\Text_Cat_Based_EDA\\pretrained_models')
sys.path.append('..\\..\\Text_Cat_Based_EDA\\evolutionary_algorithms')

from utils.embedding_builder import build_glove_from_pretrained,build_spanish_glove_from_pretrained

import torch
import os
from utils.custom_dataloader import CustomDataLoader
from utils.logging_custom import make_logger
from utils.file_arguments_reader import load_param_from_file
from scripts.main_gradient_based import train_model_sgd


if __name__ == "__main__":
    # Load train arguments from file
    os.chdir("../")
    wdir = os.getcwd() + "/" if not os.path.exists(
        "/home/CLUSTER/uclv_ogtoledano/doctorado/Text_Cat_Based_EDA/") else "/home/CLUSTER/uclv_ogtoledano/doctorado/Text_Cat_Based_EDA/"  # only slurm cluster
    dic_param = load_param_from_file(wdir + "scripts/arguments.txt")
    log_exp_run = make_logger(name="" + dic_param['name_log_experiments_result'])
    device = "cuda:" + str(dic_param['cuda_device_id']) if torch.cuda.is_available() else "cpu"

    # Load pre-trained word embedding model with specific language: Spanish or English
    tensor_embedding = build_spanish_glove_from_pretrained(wdir + 'utils/pretrained_models',
                                                           wdir + 'datasets/' + dic_param['dataset_dictionary']) if \
    dic_param['word_embedding_pretrained_glove_language'] == 'Spanish' \
        else build_glove_from_pretrained(wdir + 'utils/pretrained_models',
                                         wdir + 'datasets/' + dic_param['dataset_dictionary'])

    # Create lazy Dataloader from Tensor dataset
    train_data = CustomDataLoader(wdir + 'datasets/' + dic_param['dataset_train'])
    test_data = CustomDataLoader(wdir + 'datasets/' + dic_param['dataset_test'])

    gscv_best_model = None
    train_model_sgd(dic_param, log_exp_run, wdir, device, tensor_embedding, train_data, test_data, gscv_best_model)
