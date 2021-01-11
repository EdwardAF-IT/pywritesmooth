import logging as log
from .TrainerInterface import TrainerInterface

import numpy as np
import pandas as pd
pd.options.display.float_format = '{:,.5f}'.format

from IPython.display import display

# Sklearn tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Neural Networks
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger

# Plotting
import matplotlib.pyplot as plt

class LSTMTrainer(TrainerInterface):
    """description of class"""

    def __init__(self, params = None):
        print("In ltsm con")
        if params == None:
            self.p = dict(
                            seq_len = 24,
                            batch_size = 70, 
                            criterion = nn.MSELoss(),
                            max_epochs = 10,
                            n_features = 7,
                            hidden_size = 100,
                            num_layers = 1,
                            dropout = 0.2,
                            learning_rate = 0.001,
                        )

    def train(self, hw):
        trainer = Trainer(
            max_epochs=self.p['max_epochs'],
            logger=csv_logger,
            gpus=1,
            row_log_interval=1,
            progress_bar_refresh_rate=2,
        )

        model = LSTMRegressor(
            n_features = self.p['n_features'],
            hidden_size = self.p['hidden_size'],
            seq_len = self.p['seq_len'],
            batch_size = self.p['batch_size'],
            criterion = self.p['criterion'],
            num_layers = self.p['num_layers'],
            dropout = self.p['dropout'],
            learning_rate = self.p['learning_rate']
        )

        dm = PowerConsumptionDataModule(
            seq_len = self.p['seq_len'],
            batch_size = self.p['batch_size']
        )

        trainer.fit(model, dm)
        trainer.test(model, datamodule=dm)

    def test():
        raise NotImplementedError

    def getError():
        raise NotImplementedError

    def save():
        raise NotImplementedError

    def load():
        raise NotImplementedError