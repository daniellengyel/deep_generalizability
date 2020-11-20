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



def entropy(probs):
    return - np.sum([p * np.log(p) for p in probs])

def get_entropy(net, data, device=None):
    inputs, labels = data

    outputs = get_model_outputs(net, data, softmax_outputs=True, device=device)
    outputs = outputs.detach().numpy()
    
    return [entropy(p) for p in outputs]

# def get_max_margin(net, datapoint):
#     inp, l = datapoint
    
#     loss = criterion(outputs, labels)
#     loss.backward(retain_graph=True)
    
#     param_grads = get_grad_params_vec(net)
#     curr_weight = torch.norm(param_grads)
        
# def get_nu(data):
#     tr = 0
#     for i in range(len(data[0])):
#         inputs, labels = data[0][i], data[1][i]
#         mean_inp = torch.mean(inputs)

#         for j in range()

def get_linear_loss_trace(models, data, loss_type, device=None):

    results_filters = {}
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
            correct_filter = get_correct_filter(m, data, is_binary_classification=False)
            curr_results = 2 * (inputs_norm**2 + 1) # no dependance on number of classes since we use reduction=mean
        elif loss_type == "BinaryExponentialLoss":
            curr_point_loss, correct_filter = point_loss_filters[k]
            curr_results = (inputs_norm**2 + 1) * curr_point_loss
        elif loss_type == "cross-entropy":
            outputs = get_model_outputs(m, data, softmax_outputs=True, device=device)
            _, predicted = torch.max(outputs, 1)
            correct_filter = predicted == labels 
            correct_filter = correct_filter.detach().numpy()
            curr_results =  (inputs_norm**2 + 1) * (1 - torch.norm(outputs, dim=1)**2)
        else:
            raise NotImplementedError("Loss type {} is not implemented.".format(loss_type))

        curr_results = curr_results.detach().numpy()
        
        results_filters[k] = (curr_results, correct_filter)
    return results_filters

    
    

def get_margins_filters(models, data, device=None, get_upperbound=False, softmax_outputs=False, seed=None):
    set_seed(seed)

    margins_filters = {}
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
        correct_filter = correct_filter.detach().numpy()
        
        margins_filters[k] = (curr_margins, correct_filter)
    return margins_filters
    

def cutoffs(m1, m2, t1, t2):
    print("Trace sum first: {}".format(np.sum(t1)))
    print("Trace sum second: {}".format(np.sum(t2)))

    print()
    min_cutoff = min(min(m1), min(m2))
    max_cutoff = max(max(m1), max(m2))
    for cutoff in np.linspace(min_cutoff, max_cutoff, 20):
        print("Cutoff: {}".format(cutoff))
        f1 = m1 < cutoff
        t1 = np.array(t1)
        m1 = np.array(m1)
        s1 = np.sum(f1)
        print("Num Points first: {}".format(s1))
        print("Mean Margin first: {}".format(np.mean(m1[f1])))
        print("Mean Trace first: {}".format(np.mean(t1[f1])))

        f2 = m2 < cutoff
        t2 = np.array(t2)
        m2 = np.array(m2)
        s2 = np.sum(f2)
        print("Num Points second: {}".format(s2))
        print("Mean Margin second: {}".format(np.mean(m2[f2])))
        print("Mean Trace second: {}".format(np.mean(t2[f2])))



        print()

def get_inp_out_jacobian_points(models, data, device=None):
    out_dim = 10
    
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
 
            
        correct_filter = get_correct_filter(m, data, device=device)
        
        results_filters[k] = (np.array(point_jacobs).reshape(-1), correct_filter)
        
    return results_filters


if __name__ == "__main__":
    pass
