from skorch.callbacks import Callback
from utils.overfit_exception import Overfit_Exception
from utils.logging_custom import make_logger



class EarlyStopping(Callback):

    def __init__(self,min_diference):
        self.min_diference=min_diference # 2.4e-7

    def initialize(self):
        self.cont_epoch = 0
        self.loss_all = []
        self.cont = 10
        pass

    def on_epoch_end(self, net,**kwargs):
        loss = net.history[-1,'train_loss']
        self.loss_all.append(loss)
        self.cont_epoch += 1

        # early stoping
        if len(self.loss_all) > 1:
            if abs(self.loss_all[self.cont_epoch - 1] - self.loss_all[self.cont_epoch - 2]) < self.min_diference:
                self.cont -= 1
            else:
                self.cont = 10

        if self.cont == 0:
            log_exp_run = make_logger()
            log_exp_run.experiments(self.loss_all)
            raise Overfit_Exception()