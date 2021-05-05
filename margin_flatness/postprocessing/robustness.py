import pandas as pd
import pickle,os, copy
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import sys


from ..nets import Nets
from ..utils import get_correct_filter, get_model_outputs, set_seed, take_slice
from ..training_utils import get_criterion

def get_max_output(models, data, device=None, get_upperbound=False, softmax_outputs=False, seed=None):
    assert (not get_upperbound) or softmax_outputs # if get_upperbound then softmax_outputs

    set_seed(seed)

    max_outputs = {}
    if device is not None:
        is_gpu = True
    else:
        is_gpu = False

    inputs, labels = iter(DataLoader(data, batch_size=len(data), shuffle=False)).next()

    for k, m in models.items():

        outputs = get_model_outputs(m, data, softmax_outputs=softmax_outputs, device=device)
        _, predicted = torch.max(outputs, 1)
        correct_filter = predicted == labels 

        max_outputs[k] = torch.topk(outputs, k=1, dim=1)[0].detach().numpy()

    return max_outputs

def get_margins(models, data, device=None, get_upperbound=False, softmax_outputs=False, seed=None):
    assert (not get_upperbound) or softmax_outputs # if get_upperbound then softmax_outputs

    set_seed(seed)

    margins = {}
    if device is not None:
        is_gpu = True
    else:
        is_gpu = False

    inputs, labels = iter(DataLoader(data, batch_size=len(data), shuffle=False)).next()

    for k, m in models.items():

        outputs = get_model_outputs(m, data, softmax_outputs=softmax_outputs, device=device)
        _, predicted = torch.max(outputs, 1)
        correct_filter = predicted == labels 

        # if correctly predicted
        second_largest = torch.topk(outputs, k=2, dim=1)

        curr_margins = second_largest[0][:, 0] - second_largest[0][:, 1]
        
        # we override the ones which were incorrectly predicted
        curr_margins[~correct_filter] = second_largest[0][:, 0][~correct_filter] - torch.Tensor(take_slice(outputs, labels))[~correct_filter]
        
        curr_margins = curr_margins.detach().numpy()

        if get_upperbound:
            curr_margins = 1 - curr_margins**2

        margins[k] = curr_margins

    return margins
    

def get_inp_out_jacobian_points(models, data, device=None):
    
    results_filters = {}
    if device is not None:
        is_gpu = True
    else:
        is_gpu = False

    data_loader = DataLoader(data, batch_size=len(data), shuffle=False)

    for k, m in models.items():
        m.eval()
        
        point_jacobs = []

        for i, (inputs, labels) in enumerate(data_loader):

            # Compute gradients for input.
            curr_jacob_norm = np.zeros(len(inputs))
            out_dim = 1 # we use the first time we compute outputs to figure out what out_dim is. 
            o_idx = 0
            while o_idx < out_dim:
                inputs.requires_grad = True

                m.zero_grad()
            
                outputs = m(inputs) # get softmax of this 
                out_dim = outputs.shape[1]
                soft_layer = torch.nn.Softmax(dim=-1)
                outputs = soft_layer(outputs)
                
                torch.sum(outputs, 0)[o_idx].backward() # the trick is that the ith input vector will have derivative zero with jth output

                curr_input_grads = inputs.grad
                curr_input_grads = curr_input_grads.view(curr_input_grads.shape[0], -1).detach().numpy()


                curr_jacob_norm += np.linalg.norm(curr_input_grads, axis=1)**2   

                o_idx += 1       

            point_jacobs.append(np.sqrt(curr_jacob_norm))
 
                
        results_filters[k] = np.array(point_jacobs).reshape(-1)
        
    return results_filters

def sample_average_robustness_pointwise(models, data, criterion, N, delta, seed=None):
    set_seed(seed)
    robustness = {}
    # if device is not None:
    #     is_gpu = True
    # else:
    #     is_gpu = False

    dataloader = DataLoader(data, batch_size=1, shuffle=False) 

    for k, m in models.items():
        m.eval()

        run_sums = np.zeros(len(dataloader))

        for i, (inputs, labels) in enumerate(dataloader):

            for _ in range(N):

            # if device is not None:
            #     inputs, labels = inputs.to(device).type(torch.cuda.FloatTensor), labels.to(device).type(
            #         torch.cuda.LongTensor)
                perturb = delta * torch.rand(inputs.shape)
                outputs = m(inputs + perturb)
                run_sums[i] += criterion(outputs, labels)

        average_loss = run_sums/float(N)

        # get unpreturbed loss
        curr_loss = np.zeros(len(dataloader))
        for i, (inputs, labels) in enumerate(dataloader):

            # if device is not None:
            #     inputs, labels = inputs.to(device).type(torch.cuda.FloatTensor), labels.to(device).type(
            #         torch.cuda.LongTensor)

                outputs = m(inputs)
                curr_loss[i] += criterion(outputs, labels) 

        robustness[k] = average_loss - curr_loss
    return robustness


if __name__ == "__main__":
    pass
