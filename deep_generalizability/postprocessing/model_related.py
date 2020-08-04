import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision

from sklearn.manifold import TSNE

from .utils import *

import yaml, os, sys, re

import torch
from hessian_eigenthings import compute_hessian_eigenthings

import pickle


def get_models_loss_acc(models, train_loader, test_loader, device=None):
    loss_dict = {}
    acc_dict = {}

    for k, m in models.items():
        if device is not None:
            m = m.to(device)
        loss_dict[k] = (get_net_loss(m, train_loader, device=device), get_net_loss(m, test_loader, device=device))
        acc_dict[k] = (get_net_accuracy(m, train_loader, device=device), get_net_accuracy(m, test_loader, device=device))
    return loss_dict, acc_dict



def get_models_grad(models, data, device=None):
    grad_dict = {}

    data_loader = DataLoader(data, batch_size=len(data), shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()


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
