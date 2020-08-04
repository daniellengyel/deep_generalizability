import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision

from sklearn.manifold import TSNE

from ..utils import *
from ..data_getters import *

import yaml, os, sys, re, time

from ..pyhessian import hessian

import torch
from hessian_eigenthings import compute_hessian_eigenthings

import pickle


def get_point_traces(models, data, device=None):
    criterion = torch.nn.CrossEntropyLoss()
    traces = {}
    if device is not None:
        is_gpu = True
    else:
        is_gpu = False

    for k, m in models.items():
        curr_traces = []
        for i in range(len(data[0])):
            inputs, labels = data[0][i], data[1][i]
            inputs, labels = inputs.view(1, *inputs.shape), labels.view(1, *labels.shape)

            if device is not None:
                inputs, labels = inputs.to(device).type(torch.cuda.FloatTensor), labels.to(device).type(
                    torch.cuda.LongTensor)

            curr_traces.append(np.mean(hessian(m, criterion, data=(inputs, labels), cuda=is_gpu).trace(maxIter=1000)))
        traces[k] = curr_traces
    return traces

def get_point_eig_density_traces(models, data, device=None):
    criterion = torch.nn.CrossEntropyLoss()
    traces = {}
    if device is not None:
        is_gpu = True
    else:
        is_gpu = False

    for k, m in models.items():
        curr_traces = []
        for i in range(len(data[0])):
            inputs, labels = data[0][i], data[1][i]
            inputs, labels = inputs.view(1, *inputs.shape), labels.view(1, *labels.shape)

            if device is not None:
                inputs, labels = inputs.to(device).type(torch.cuda.FloatTensor), labels.to(device).type(
                    torch.cuda.LongTensor)

            eigs, density = hessian(m, criterion, data=(inputs, labels), cuda=is_gpu).density(iter=100, n_v=1)
            curr_traces.append(np.array(eigs[0]).dot(np.array(density[0])))
        traces[k] = curr_traces
    return traces

def compute_trace_from_eig_density(exp_eig_density_dict):
    traces = {}
    for exp_id in exp_eig_density_dict:
        traces[exp_id] = {}
        for model_idx in exp_eig_density_dict[exp_id]:
            traces[exp_id][model_idx] = []
            for point_eig_density in exp_eig_density_dict[exp_id][model_idx]:
                eigs, density = point_eig_density
                point_trace = np.array(eigs[0]).dot(np.array(density[0]))
                traces[exp_id][model_idx].append(point_trace)
            traces[exp_id][model_idx] = np.array(traces[exp_id][model_idx])
    return traces

def get_point_eig_density(models, data, device=None):
    criterion = torch.nn.CrossEntropyLoss()
    eig_density = {}
    if device is not None:
        is_gpu = True
    else:
        is_gpu = False

    for k, m in models.items():
        curr_eig_density = []
        for i in range(len(data[0])):
            inputs, labels = data[0][i], data[1][i]
            inputs, labels = inputs.view(1, *inputs.shape), labels.view(1, *labels.shape)

            if device is not None:
                inputs, labels = inputs.to(device).type(torch.cuda.FloatTensor), labels.to(device).type(
                    torch.cuda.LongTensor)

            curr_eig_density.append(hessian(m, criterion, data=(inputs, labels), cuda=is_gpu).density(iter=10, n_v=1))
        eig_density[k] = curr_eig_density
    return eig_density

# get eigenvalues of specific model folder.
def get_models_eig(models, train_loader, test_loader, loss, num_eigenthings=5, full_dataset=True, device=None, only_vals=True):
    eig_dict = {}
    # get eigenvals
    for k, m in models.items():
        print(k)
        if device is not  None:
            m = m.to(device)
            is_gpu = True
        else:
            is_gpu = False

        eigenvals, eigenvecs = compute_hessian_eigenthings(m, train_loader,
                                                           loss, num_eigenthings, use_gpu=is_gpu,
                                                           full_dataset=full_dataset, mode="lanczos",
                                                           max_steps=100, tol=1e-2)
        try:
            #     eigenvals, eigenvecs = compute_hessian_eigenthings(m, train_loader,
            #                                                        loss, num_eigenthings, use_gpu=use_gpu, full_dataset=full_dataset , mode="lanczos",
            #                                                        max_steps=50)
            if only_vals:
                eig_dict[k] = eigenvals
            else:
                eig_dict[k] = (eigenvals, eigenvecs)
        except:
            print("Error for net {}.".format(k))

    return eig_dict



def get_models_trace(models, data_loader, criterion, full_dataset=False, verbose=False, device=None):
    trace_dict = {}

    hessian_dataloader = []
    for i, (inputs, labels) in enumerate(data_loader):
        hessian_dataloader.append((inputs, labels))
        if not full_dataset:
            break

    # get trace
    for k, m in models.items():
        if verbose:
            print(k)
        a = time.time()
        ts = []

        if device is not  None:
            m = m.to(device)
            is_gpu = True
        else:
            is_gpu = False

        if full_dataset:
            trace = hessian(m, criterion, dataloader=hessian_dataloader, cuda=is_gpu).trace()
        else:
            trace = hessian(m, criterion, data=hessian_dataloader[0], cuda=is_gpu).trace()

        trace_dict[k] = trace

    return trace_dict
