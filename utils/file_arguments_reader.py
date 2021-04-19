# Load parameters from file
def load_param_from_file(url_param):
    param_file = url_param
    sample = open(url_param, "r")
    params = sample.readlines()
    dictionary_param = {}
    for values in params:
        aux = values.split('=')

        if aux[0] in ['dataset_train', 'dataset_test', 'dataset_dictionary', 'cnn_checkpoint', 'name_log_experiments_result','f_params_name','f_optimizer_name','f_history_name','cnn_optimizer','f_criterion_name','word_embedding_pretrained_glove_language']:
            dictionary_param[aux[0]] = aux[1].split('\n')[0]

        if aux[0] in ['epochs','epochs_gs_cv','num_executions','cuda_device_id', 'generations', 'sgd_early_stopping_patientia', 'sgd_batch_size', 'grid_search_cross_val_cv', 'individuals', 'labels','word_embedding_size']:
            dictionary_param[aux[0]] = int(aux[1].split('\n')[0])

        if aux[0] in ['sgd_min_difference', "l2_reg", 'sigma', 'centroid']:
            dictionary_param[aux[0]] = float(aux[1].split('\n')[0])

        if aux[0] in ['alpha_distribution', 'centroid_distribution', 'sigma_distribution','momentum_distribution']:
            list_values = aux[1].split('\n')[0].split(',')
            array_values = []

            for elem in list_values:
                array_values.append(float(elem))

            dictionary_param[aux[0]] = array_values

    return dictionary_param
