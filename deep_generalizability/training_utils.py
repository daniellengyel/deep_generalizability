import numpy as np
import copy, yaml, pickle
import torch
import torchvision
import torch.optim as optim

from .utils import *
from .nets.Nets import SimpleNet, LeNet

from torch.utils.data import DataLoader
import os, time


def get_nets(net_name, net_params, num_nets, device=None):
    if net_name == "SimpleNet":
        nets = [SimpleNet(*net_params) for _ in range(num_nets)]
    elif net_name == "LeNet":
        nets = [LeNet(*net_params) for _ in range(num_nets)]
    else:
        raise NotImplementedError("{} is not implemented.".format(net_name))
    if device is not None:
        nets = [net.to(device) for net in nets]
    return nets

def get_optimizers(config):
    def helper(nets, optimizers=None):
        num_nets = config["num_nets"]

        if optimizers is None:
            # optimizers = [optim.Adam(nets[i].parameters(), lr=config["learning_rate"]) for i in range(num_nets)]
            optimizers = [optim.SGD(nets[i].parameters(), lr=config["learning_rate"],
                                    momentum=config["momentum"]) for i in range(num_nets)]
            if ("lr_decay" in config) and (config["lr_decay"] is not None) and (config["lr_decay"] != 0):
                optimizers = [torch.optim.lr_scheduler.ExponentialLR(optimizer=o, gamma=config["lr_decay"]) for o in
                              optimizers]
        else:
            # update opt with new nets
            for i in range(len(optimizers)):
                o = optimizers[i]
                nn = nets[i]
                o.param_groups = []

                param_groups = list(nn.parameters())
                if len(param_groups) == 0:
                    raise ValueError("optimizer got an empty parameter list")
                if not isinstance(param_groups[0], dict):
                    param_groups = [{'params': param_groups}]

                for param_group in param_groups:
                    o.add_param_group(param_group)

        return optimizers
    return helper

def get_resampling_checker(sampling_tau, sampling_wait, sampling_stop, ess_threshold, beta):
    if sampling_stop is None:
        sampling_stop = float("inf")
    if sampling_wait is None:
        sampling_wait = 0
    assert not ((ess_threshold is not None) and (sampling_tau is not None))
    if ess_threshold is not None:
        should_resample = lambda w, s: (kish_effs(w) < ess_threshold) and (beta != 0) and (s >= sampling_wait)
    elif sampling_tau is not None:
        should_resample = lambda w, s: (s % sampling_tau == 0) and (s > 0) and (s >= sampling_wait) and (
                beta != 0) and (s < sampling_stop)
    else:
        should_resample = lambda w, s: False
    return should_resample

def get_stopping_criterion(num_steps, mean_loss_threshold):
    if (num_steps is not None) and (mean_loss_threshold is not None):
        stopping_criterion = lambda ml, s: (num_steps < s) or (ml < mean_loss_threshold)
    elif num_steps is not None:
        stopping_criterion = lambda ml, s: num_steps < s
    elif mean_loss_threshold is not None:
        stopping_criterion = lambda ml, s: ml < mean_loss_threshold
    else:
        raise Exception("Error: Did not provide a stopping criterion.")
    return stopping_criterion

def add_noise(net, var_noise, device=None):
    with torch.no_grad():
        for param in net.parameters():
            noise = torch.randn(param.size()) * var_noise
            if device is not None:
                noise = noise.to(device)
            param.add_(noise)
    return net

