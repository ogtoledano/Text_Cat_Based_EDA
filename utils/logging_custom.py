# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------+
#
# @author: Doctorini
# This module creates a custom logging for making experiments
# with log level: EXPERIMENTS_RESULTS
# The main results are saved into experiments folder, on the project by default
# Log file generate is named as experiment.log  
#
#------------------------------------------------------------------------------+

import logging
import os

EXPERIMENTS_RESULTS_LOG_LEVEL = 9

# Registring the own log module
logging.addLevelName(EXPERIMENTS_RESULTS_LOG_LEVEL, 'EXPERIMENTS_RESULTS')


# Defining the own log module
def experiments(self, message,*args,**kws):
    self._log(EXPERIMENTS_RESULTS_LOG_LEVEL,message,args,**kws)


# Return the main instance for handling logging
def make_logger(name="experiment"):
    logging.Logger.experiments = experiments

    # Main path and file for logging
    #os.chdir("../")
    wdir = os.getcwd() + "/" if not os.path.exists("/home/CLUSTER/uclv_ogtoledano/doctorado/Text_Cat_Based_EDA/") else "/home/CLUSTER/uclv_ogtoledano/doctorado/Text_Cat_Based_EDA/"  # only slurm cluster
    urlPath = wdir + "/experiments"  # ../experiments
    logging.basicConfig(filename=urlPath + "/" + name + ".log")

    # Turn off the auto-logging generation by Python modules, such as: "gensim" module
    logging.getLogger("gensim").setLevel(logging.CRITICAL)  # Only generates CRITICAL logs for "gensim" module
    logging.getLogger("torch").setLevel(logging.CRITICAL)  # Only generates CRITICAL logs for "torch" module

    log_exp=logging.getLogger()
    return log_exp
