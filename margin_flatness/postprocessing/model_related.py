import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision

from sklearn.manifold import TSNE

from .utils import *
from ..utils import *
from ..training_utils import get_criterion

import yaml, os, sys, re

import torch

import pickle


def get_models_loss_acc(models, train_data, test_data, criterion, loss_type, device=None, seed=None):
    set_seed(seed)

    loss_dict = {}
    acc_dict = {}

    train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    is_binary_classification = loss_type == "BinaryExponentialLoss"

    for k, m in models.items():
        if device is not None:
            m = m.to(device)
        loss_dict[k] = (get_net_loss(m, train_loader, criterion, device=device), get_net_loss(m, test_loader, criterion, device=device))
        acc_dict[k] = (get_net_accuracy(m, train_loader, is_binary_classification, device=device), get_net_accuracy(m, test_loader, is_binary_classification, device=device))
    return loss_dict, acc_dict

def get_point_loss(models, data, loss_type, device=None):

    results = {}
    if device is not None:
        is_gpu = True
    else:
        is_gpu = False

    data_loader = DataLoader(data, batch_size=1, shuffle=False)
    criterion = get_criterion(loss_type=loss_type)
    is_binary_classification = loss_type in ["BinaryExponentialLoss"]

    for k, m in models.items():
        
        point_losses = []
        for i, (inputs, labels) in enumerate(data_loader):
            outputs = m(inputs)
            point_losses.append(float(criterion(outputs, labels)))
        point_losses = np.array(point_losses)
        
        results[k] = point_losses
    return results

def get_models_grad(models, data, criterion, device=None):
    grad_dict = {}

    data_loader = DataLoader(data, batch_size=len(data), shuffle=False)


    # get trace
    for k, m in models.items():
        curr_grads = []
        # for i, (inputs, labels) in enumerate(data_loader):
        for i in range(len(data[0])):
            inputs, labels = data[0][i], data[1][i]
            inputs, labels = inputs.view(1, *inputs.shape), labels.view(1, *labels.shape)

            if device is not None:
                    inputs, labels = inputs.to(device).type(torch.cuda.FloatTensor), labels.to(device).type(
                        torch.cuda.LongTensor)

            # Compute gradients for input.
            inputs.requires_grad = True

            m.zero_grad()

            outputs = m(inputs)
            loss = criterion(outputs.float(), labels)
            loss.backward(retain_graph=True)

            param_grads = get_grad_params_vec(m)
            curr_grads.append(float(torch.norm(param_grads)))


        grad_dict[k] = np.array(curr_grads)

    return grad_dict


def get_models_tsne(models):
    models_vecs = np.array(
        [get_params_vec(m).detach().numpy() for k, m in sorted(models.items(), key=lambda item: int(item[0]))])
    X_embedded = TSNE(n_components=2).fit_transform(models_vecs)
    return X_embedded


def get_models_final_distances(beginning_models, final_models):
    dist_arr = []
    for i in range(len(beginning_models)):
        b_vec = get_params_vec(beginning_models[str(i)])
        f_vec = get_params_vec(final_models[str(i)])
        dist_arr.append(float(torch.norm(b_vec - f_vec)))

    return dist_arr

def get_inp_out_jacobian(models, data, criterion, device=None):
    jacob_dict = {}

    data_loader = DataLoader(data, batch_size=len(data), shuffle=False)


    # get trace
    for k, m in models.items():
        curr_jacs = []
        # for i, (inputs, labels) in enumerate(data_loader):
        for i in range(len(data[0])):
            inputs, labels = data[0][i], data[1][i]
            inputs, labels = inputs.view(1, *inputs.shape), labels.view(1, *labels.shape)

            if device is not None:
                    inputs, labels = inputs.to(device).type(torch.cuda.FloatTensor), labels.to(device).type(
                        torch.cuda.LongTensor)

            # Compute gradients for input.
            inputs.requires_grad = True

            m.zero_grad()

            outputs = m(inputs)

            # zero the parameter

            curr_jacob_norm = 0
            for i in range(len(inputs)):
                for o in outputs:
                    m.zero_grad()

                    o.backward()

                    curr_input_grads = inputs.grad
                    curr_input_grads = grad.view((input_shape[0], np.product(input_shape[1:]))) # batch, rest
                    
                    curr_jacob_norm += torch.norm(curr_input_grads, axis=1)**2            

            curr_jacs.append(curr_jacob_norm.detach().numpy())


        jacob_dict[k] = np.mean(np.array(curr_jacs).flatten())

    return jacob_dict