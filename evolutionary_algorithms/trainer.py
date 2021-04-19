import sys
sys.path.append('..\\..\\Text_Cat_Based_EDA')
sys.path.append('..\\..\\Text_Cat_Based_EDA\\utils')
sys.path.append('..\\..\\Text_Cat_Based_EDA\\pretrained_models')
sys.path.append('..\\..\\Text_Cat_Based_EDA\\evolutionary_algorithms')

import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.logging_custom import make_logger

# Scikit-learn ----------------------------------------------------------+
from sklearn.metrics import classification_report,accuracy_score
from skorch import NeuralNet


class Trainer(NeuralNet):

    def __init__(self,*args,mode="Adam",**kargs):
        super().__init__(*args, **kargs)
        #self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode=mode
        log_exp_run = make_logger(name="experiment_" + self.mode)
        log_exp_run.experiments("Running on device: "+str(self.device))
        log_exp_run.experiments("Training model by Back-propagation with optimizer: "+mode)

    def initialize_criterion(self,*args,**kargs):
        super().initialize_criterion(*args,**kargs)
        return self

    def initialize_module(self,*args,**kargs):
        super().initialize_module(*args, **kargs)
        param_length = sum([p.numel() for p in self.module_.parameters() if p.requires_grad])
        log_exp_run = make_logger(name="experiment_" + self.mode)
        log_exp_run.experiments("Amount of parameters: " + str(param_length))
        return self

    # Sci-kit methods
    def predict(self, X):
        x_train = X['features'].type(torch.LongTensor)
        x_train = x_train.to(self.device)
        self.module_.to(self.device)
        prob = self.module_(x_train)
        _, predicted = torch.max(prob.data, 1)
        return predicted.cpu().numpy()[0]

    # Skorch methods: Compute the loss function using softmax and compute score by accuracy
    def score(self, X, y=None):
        train_loss = 0
        criterion = nn.CrossEntropyLoss()
        iter_data = DataLoader(X, batch_size=self.module__batch_size, shuffle=True)
        log_exp_run = make_logger(name="experiment_" + self.mode)

        predictions = []
        labels = []
        self.module_.to(self.device)
        self.module_.eval()

        with torch.no_grad():
            for bach in iter_data:
                x_test = bach['features'].type(torch.LongTensor)
                y_test = bach['labels'].type(torch.LongTensor)
                x_test = x_test.to(self.device)
                y_test = y_test.to(self.device)
                prob = self.module_(x_test)
                loss = criterion(prob, y_test)
                train_loss += loss.item()
                _, predicted = torch.max(prob.data, 1)
                predictions.extend(predicted.cpu().numpy())
                labels.extend(y_test.cpu().numpy())

        accuracy = accuracy_score(predictions, labels)

        log_exp_run.experiments("Cross-entropy loss for each fold: " + str(train_loss))
        log_exp_run.experiments("Accuracy for each fold: " + str(accuracy))
        log_exp_run.experiments("\n"+classification_report(labels, predictions))
        return accuracy

    # Skorch methods: this method fits the estimator by back-propagation and an optimizer
    # SGD or ADAM
    def fit(self, X, y=None, **fit_params):
        log_exp_run = make_logger(name="experiment_" + self.mode)

        if not self.warm_start or not self.initialized_:
            self.initialize()

        self.X_ = X

        train_loss_acc=[]
        self.module_.to(self.device)
        optimizer = self.optimizer_
        criterion = self.criterion_
        iter_data = DataLoader(X, batch_size=self.module__batch_size, shuffle=True)

        patientia = fit_params["patientia"] if fit_params.get('fit_param') is None else fit_params["fit_param"]["patientia"]
        cont_early_stoping = fit_params["patientia"] if fit_params.get('fit_param') is None else fit_params["fit_param"]["patientia"]
        min_diference = fit_params["min_diference"] if fit_params.get('fit_param') is None else fit_params["fit_param"]["min_diference"]
        self.notify('on_train_begin', X=X, y=y)

        isinstance(optimizer,Adam)
        log_exp_run.experiments("Run using {} as optimizer".format("Adam" if isinstance(optimizer,Adam) else "SGD"))

        if isinstance(optimizer, Adam):
            log_exp_run.experiments("lr: {}".format(self.lr))
        else:
            log_exp_run.experiments("lr: {} and momentum: {}".format(self.lr, self.optimizer__momentum))

        on_epoch_kwargs = {
            'dataset_train': X,
            'dataset_valid': None,
        }

        for epoch in range(self.max_epochs):
            train_loss = 0
            self.notify('on_epoch_begin',**on_epoch_kwargs)
            for bach in iter_data:
                self.module_.zero_grad()
                x_train = bach['features'].type(torch.LongTensor)
                y_train = bach['labels'].type(torch.LongTensor)
                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)
                self.notify("on_batch_begin", X=x_train, y=y_train, training=True)
                prob = self.module_(x_train)
                loss = criterion(prob, y_train)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                self.notify("on_batch_end", X=x_train, y=y_train, training=True)

            log_exp_run.experiments("Epoch ran: " + str(epoch) + " loss: " + str(train_loss))
            train_loss_acc.append(train_loss)

            self.notify('on_epoch_end',**on_epoch_kwargs)
            if len(train_loss_acc) > 1:
                if abs(train_loss_acc[epoch - 1] - train_loss_acc[epoch - 2]) < min_diference:
                    cont_early_stoping -= 1
                else:
                    cont_early_stoping = patientia

            if cont_early_stoping == 0:
                break

        log_exp_run.experiments("Train loss series:")
        log_exp_run.experiments(train_loss_acc)
        self.notify('on_train_end', X=X, y=y)
        return self
