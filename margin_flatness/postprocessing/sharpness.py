import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision

from sklearn.manifold import TSNE

from ..utils import *
from ..data_getters import *
from ..nets.Nets import UnitvectorOutputNet

import yaml, os, sys, re, time

from ..pyhessian import hessian

import torch
# from hessian_eigenthings import compute_hessian_eigenthings

import pickle



def get_point_traces(models, data, criterion, device=None, seed=None):
    set_seed(seed)

    traces = {}
    if device is not None:
        is_gpu = True
    else:
        is_gpu = False

    dataloader = DataLoader(data, batch_size=1, shuffle=False) 

    for k, m in models.items():
        curr_traces = []
        for i, (inputs, labels) in enumerate(dataloader):

            if device is not None:
                inputs, labels = inputs.to(device).type(torch.cuda.FloatTensor), labels.to(device).type(
                    torch.cuda.LongTensor)

            curr_traces.append(np.mean(hessian(m, criterion, data=(inputs, labels), cuda=is_gpu).trace(maxIter=100))) # TODO: MEAN OR NOT TO MEAN
        traces[k] = curr_traces
    return traces

def get_point_unit_traces(models, data, criterion, device=None, seed=None):
    set_seed(seed)

    traces = {}
    if device is not None:
        is_gpu = True
    else:
        is_gpu = False

    dataloader = DataLoader(data, batch_size=1, shuffle=False) 

    for k, m in models.items():
        curr_traces = []
        for i, (inputs, labels) in enumerate(dataloader):

            if device is not None:
                inputs, labels = inputs.to(device).type(torch.cuda.FloatTensor), labels.to(device).type(
                    torch.cuda.LongTensor)

            curr_traces.append(np.mean(hessian(UnitvectorOutputNet(m), criterion, data=(inputs, labels), cuda=is_gpu).trace(maxIter=100))) # TODO: MEAN OR NOT TO MEAN
        traces[k] = curr_traces
    return traces

def get_affine_trace(models, data, loss_type, device=None):

    results = {}
    if device is not None:
        is_gpu = True
    else:
        is_gpu = False

    inputs, labels = iter(DataLoader(data, batch_size=len(data), shuffle=False)).next()
    inputs_norm = torch.norm(inputs.view(inputs.shape[0], -1), dim=1)

    if loss_type == "BinaryExponentialLoss":
        point_loss_filters = get_point_loss_filters(models, data, loss_type, device=device)

    for k, m in models.items():


        if loss_type == "MSE":
            curr_results = 2 * (inputs_norm**2 + 1) # no dependance on number of classes since we use reduction=mean
        elif loss_type == "BinaryExponentialLoss":
            curr_point_loss, correct_filter = point_loss_filters[k]
            curr_results = (inputs_norm**2 + 1) * curr_point_loss
        elif loss_type == "cross-entropy":
            outputs = get_model_outputs(m, data, softmax_outputs=True, device=device)
            curr_results =  (inputs_norm**2 + 1) * (1 - torch.norm(outputs, dim=1)**2)
        else:
            raise NotImplementedError("Loss type {} is not implemented.".format(loss_type))

        curr_results = curr_results.detach().numpy()
        
        results[k] = curr_results
    return results

# TODO: What to do about negative values
def sample_average_flatness_pointwise(models, data, criterion, meta, seed=None):
    set_seed(seed)
    flatness = {}
    # if device is not None:
    #     is_gpu = True
    # else:
    #     is_gpu = False

    N, delta = meta["N"], meta["delta"]

    dataloader = DataLoader(data, batch_size=len(data), shuffle=False) 

    for k, m in models.items():
        m.eval()

        run_sums = np.zeros(len(data))
        vec_net = get_params_vec(m)

        for _ in range(N):
            perturbed_net = vec_to_net(vec_net + delta*torch.rand(len(vec_net)), m) # TODO different size

            for i, (inputs, labels) in enumerate(dataloader):

            # if device is not None:
            #     inputs, labels = inputs.to(device).type(torch.cuda.FloatTensor), labels.to(device).type(
            #         torch.cuda.LongTensor)

                outputs = perturbed_net(inputs)
                for j, o in enumerate(outputs):
                    run_sums[j] += criterion(o.view(1, -1), labels[j].view(1))
        average_loss = run_sums/float(N)

        # get unpreturbed loss
        curr_loss = np.zeros(len(data))
        for i, (inputs, labels) in enumerate(dataloader):

            # if device is not None:
            #     inputs, labels = inputs.to(device).type(torch.cuda.FloatTensor), labels.to(device).type(
            #         torch.cuda.LongTensor)

            outputs = m(inputs)
            for j, o in enumerate(outputs):
                curr_loss[j] += criterion(o.view(1, -1), labels[j].view(1))

        flatness[k] = average_loss - curr_loss
    return flatness

# Slower version but more accurate since i am not reusing the preturbed net for other datapoints. 
    # for k, m in models.items():
    #     m.eval()
    #     curr_flatness = []
    #     for i, (inputs, labels) in enumerate(dataloader):

    #         # if device is not None:
    #         #     inputs, labels = inputs.to(device).type(torch.cuda.FloatTensor), labels.to(device).type(
    #         #         torch.cuda.LongTensor)
    #         run_sum = 0
    #         for _ in range(N):
    #             vec_net = get_params_vec(m)
    #             perturbed_net = vec_to_net(vec_net + delta*torch.rand(len(vec_net)), m) # TODO different size
    #             outputs = perturbed_net(inputs)
    #             run_sum += criterion(outputs, labels)
    #         curr_flatness.append(run_sum/float(N))
    #     flatness[k] = curr_flatness
    # return flatness

def get_point_eig_density_traces(models, data, criterion, device=None, seed=None):
    set_seed(seed)

    traces = {}
    if device is not None:
        is_gpu = True
    else:
        is_gpu = False

    dataloader = DataLoader(data, batch_size=1, shuffle=False) 

    for k, m in models.items():
        curr_traces = []
        for i, (inputs, labels) in enumerate(dataloader):

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

def get_point_eig_density(models, data, criterion, device=None, seed=None):
    set_seed(seed)

    eig_density = {}
    if device is not None:
        is_gpu = True
    else:
        is_gpu = False

    dataloader = DataLoader(data, batch_size=1, shuffle=False) 

    for k, m in models.items():
        curr_eig_density = []
        for i, (inputs, labels) in enumerate(dataloader):

            if device is not None:
                inputs, labels = inputs.to(device).type(torch.cuda.FloatTensor), labels.to(device).type(
                    torch.cuda.LongTensor)

            curr_eig_density.append(hessian(m, criterion, data=(inputs, labels), cuda=is_gpu).density(iter=10, n_v=1))
        eig_density[k] = curr_eig_density
    return eig_density

# # get eigenvalues of specific model folder.
# def get_models_eig(models, train_loader, test_loader, criterion, num_eigenthings=5, full_dataset=True, device=None, only_vals=True, seed=None):
#     set_seed(seed)

#     eig_dict = {}
#     # get eigenvals
#     for k, m in models.items():
#         print(k)
#         if device is not  None:
#             m = m.to(device)
#             is_gpu = True
#         else:
#             is_gpu = False

#         eigenvals, eigenvecs = compute_hessian_eigenthings(m, train_loader,
#                                                            criterion, num_eigenthings, use_gpu=is_gpu,
#                                                            full_dataset=full_dataset, mode="lanczos",
#                                                            max_steps=100, tol=1e-2)
#         try:
#             #     eigenvals, eigenvecs = compute_hessian_eigenthings(m, train_loader,
#             #                                                        criterion, num_eigenthings, use_gpu=use_gpu, full_dataset=full_dataset , mode="lanczos",
#             #                                                        max_steps=50)
#             if only_vals:
#                 eig_dict[k] = eigenvals
#             else:
#                 eig_dict[k] = (eigenvals, eigenvecs)
#         except:
#             print("Error for net {}.".format(k))

#     return eig_dict



def get_models_trace(models, data_loader, criterion, full_dataset=False, verbose=False, device=None, seed=None):
    set_seed(seed)

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

