import numpy as np
import datetime
import math

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

def which_model(model_name):
    if model_name == '1.2':
        return TabularNN_v1_2, CFG_v1_2
    if model_name == '2':
        return TabularNN_v2, CFG_v2

class TabularNN_v1_2(nn.Module):
    def __init__(self, cfg,last_layer_bias=None):
        super().__init__()

        self.cfg = cfg

        self.batchnorm0 = nn.BatchNorm1d(cfg.num_features)

        self.dense1 = nn.utils.weight_norm(nn.Linear(cfg.num_features, cfg.hidden_size_layer1))
        self.prelu1 = nn.PReLU()
        self.batchnorm1 = nn.BatchNorm1d(cfg.hidden_size_layer1)
        self.dropout1 = nn.Dropout(cfg.dropout)

        self.dense2 = nn.utils.weight_norm(nn.Linear(cfg.hidden_size_layer1, cfg.hidden_size_layer2))
        self.prelu2 = nn.PReLU()
        self.batchnorm2 = nn.BatchNorm1d(cfg.hidden_size_layer2)
        self.dropout2 = nn.Dropout(cfg.dropout)

        self.dense3 = nn.utils.weight_norm(nn.Linear(cfg.hidden_size_layer2, cfg.target_cols))


        if not last_layer_bias is None:
            last_layer_bias = np.log(last_layer_bias.mean(axis=0).clamp(1e-10,1))
            self.dense3.bias.data = nn.Parameter(last_layer_bias.float())

        self.params = []
        self.params += self.set_bias_weight_decay(self.batchnorm0,none_to_zero=True)
        self.params += self.set_bias_weight_decay(self.dense1)
        self.params += self.set_bias_weight_decay(self.prelu1,all_to_zero=True)
        self.params += self.set_bias_weight_decay(self.batchnorm1,none_to_zero=True)
        self.params += self.set_bias_weight_decay(self.dropout1,all_to_zero=True)
        self.params += self.set_bias_weight_decay(self.dense2)
        self.params += self.set_bias_weight_decay(self.prelu2,all_to_zero=True)
        self.params += self.set_bias_weight_decay(self.batchnorm2,none_to_zero=True)
        self.params += self.set_bias_weight_decay(self.dropout2,all_to_zero=True)
        self.params += self.set_bias_weight_decay(self.dense3)


    def set_bias_weight_decay(self,layer,none_to_zero=False,all_to_zero=False):
        params = []
        named_params = dict(layer.named_parameters())
        for key, value in named_params.items():
            if none_to_zero:
                params += [{'params':value,'weight_decay':self.cfg.weight_decay}]
            else:
                if key == 'bias':
                    params += [{'params':value,'weight_decay':0.0}]
                else:
                    if all_to_zero:
                        params += [{'params':value,'weight_decay':0.0}]
                    else:
                        params += [{'params':value,'weight_decay':self.cfg.weight_decay}]
        return params

    def recalibrate_layer(self, layer):
        if(torch.isnan(layer.weight_v).sum() > 0):
            layer.weight_v = torch.nn.Parameter(torch.where(torch.isnan(layer.weight_v), torch.zeros_like(layer.weight_v), layer.weight_v))
            layer.weight_v = torch.nn.Parameter(layer.weight_v + 1e-7)

        if(torch.isnan(layer.weight).sum() > 0):
            layer.weight = torch.where(torch.isnan(layer.weight), torch.zeros_like(layer.weight), layer.weight)
            layer.weight += 1e-7

    def forward(self, x):
        x0 = self.batchnorm0(x)

        self.recalibrate_layer(self.dense1)
        x1 = self.dense1(x0)
        x1 = self.prelu1(x1)
        x1 = self.batchnorm1(x1)
        x1 = self.dropout1(x1)

        self.recalibrate_layer(self.dense2)
        x2 = self.dense2(x1)
        x2 = self.prelu2(x2)
        x2 = self.batchnorm2(x2)
        x2 = self.dropout2(x2)

        self.recalibrate_layer(self.dense3)
        y = self.dense3(x2)
        return y

class CFG_v1_2:
    hidden_size_layer1=2048
    hidden_size_layer2=2048
    dropout=0.4
    weight_decay=1e-5
    batch_size=128
    #epochs=120
    epochs=1
    min_epochs = 40
    one_cycle_epochs=100
    number_one_cycle=1
    early_stopping=30
    learning_rate=1e-3
    patience=30
    hard_patience=25
    min_delta=0.00005
    ratio_train_val=1.15
    pct_start=0.1
    div_factor=1e3
    verbose=1

class TabularNN_v2(nn.Module):
    def __init__(self, cfg,last_layer_bias=None):
        super().__init__()

        self.cfg = cfg

        self.batchnorm0 = nn.BatchNorm1d(cfg.num_features)

        self.dense1 = nn.utils.weight_norm(nn.Linear(cfg.num_features, cfg.hidden_size_layer1))
        self.prelu1 = nn.PReLU()
        self.batchnorm1 = nn.BatchNorm1d(cfg.hidden_size_layer1)
        self.dropout1 = nn.Dropout(cfg.dropout)

        self.dense2 = nn.utils.weight_norm(nn.Linear(cfg.hidden_size_layer1, cfg.hidden_size_layer2))
        self.prelu2 = nn.PReLU()
        self.batchnorm2 = nn.BatchNorm1d(cfg.hidden_size_layer2)
        self.dropout2 = nn.Dropout(cfg.dropout)

        self.dense3 = nn.utils.weight_norm(nn.Linear(cfg.hidden_size_layer2, cfg.target_cols))


        if not last_layer_bias is None:
            last_layer_bias = np.log(last_layer_bias.mean(axis=0).clamp(1e-10,1))
            self.dense3.bias.data = nn.Parameter(last_layer_bias.float())

        self.params = []
        self.params += self.set_bias_weight_decay(self.batchnorm0,none_to_zero=True)
        self.params += self.set_bias_weight_decay(self.dense1)
        self.params += self.set_bias_weight_decay(self.prelu1,all_to_zero=True)
        self.params += self.set_bias_weight_decay(self.batchnorm1,none_to_zero=True)
        self.params += self.set_bias_weight_decay(self.dropout1,all_to_zero=True)
        self.params += self.set_bias_weight_decay(self.dense2)
        self.params += self.set_bias_weight_decay(self.prelu2,all_to_zero=True)
        self.params += self.set_bias_weight_decay(self.batchnorm2,none_to_zero=True)
        self.params += self.set_bias_weight_decay(self.dropout2,all_to_zero=True)
        self.params += self.set_bias_weight_decay(self.dense3)


    def set_bias_weight_decay(self,layer,none_to_zero=False,all_to_zero=False):
        params = []
        named_params = dict(layer.named_parameters())
        for key, value in named_params.items():
            if none_to_zero:
                params += [{'params':value,'weight_decay':self.cfg.weight_decay}]
            else:
                if key == 'bias':
                    params += [{'params':value,'weight_decay':0.0}]
                else:
                    if all_to_zero:
                        params += [{'params':value,'weight_decay':0.0}]
                    else:
                        params += [{'params':value,'weight_decay':self.cfg.weight_decay}]
        return params

    def recalibrate_layer(self, layer):
        if(torch.isnan(layer.weight_v).sum() > 0):
            layer.weight_v = torch.nn.Parameter(torch.where(torch.isnan(layer.weight_v), torch.zeros_like(layer.weight_v), layer.weight_v))
            layer.weight_v = torch.nn.Parameter(layer.weight_v + 1e-7)

        if(torch.isnan(layer.weight).sum() > 0):
            layer.weight = torch.where(torch.isnan(layer.weight), torch.zeros_like(layer.weight), layer.weight)
            layer.weight += 1e-7

    def forward(self, x):
        x0 = self.batchnorm0(x)

        self.recalibrate_layer(self.dense1)
        x1 = self.dense1(x0)
        x1 = self.prelu1(x1)
        x1 = self.batchnorm1(x1)
        x1 = self.dropout1(x1)

        self.recalibrate_layer(self.dense2)
        x2 = self.dense2(x1)
        x2 = self.prelu2(x2)
        x2 = self.batchnorm2(x2)
        x2 = self.dropout2(x2)

        self.recalibrate_layer(self.dense3)
        y = self.dense3(x2)
        return y

class CFG_v2:
    hidden_size_layer1=1024
    hidden_size_layer2=1024
    dropout=0.4
    weight_decay=1e-5
    batch_size=128
    epochs=100
    min_epochs = 25
    one_cycle_epochs=25
    number_one_cycle=1
    early_stopping=20
    learning_rate=1e-3
    patience=15
    hard_patience=25
    min_delta=0.00005
    ratio_train_val=1.15
    pct_start=0.1
    div_factor=1e3
    verbose=1

