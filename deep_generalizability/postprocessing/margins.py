import pandas as pd
import pickle,os, copy
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import sys


from deep_generalizability.nets import Nets
from deep_generalizability.utils import *
from deep_generalizability.postprocessing.postprocessing import *
from deep_generalizability.postprocessing.stats_plotting import *
from deep_generalizability.postprocessing.GA import *
from deep_generalizability.save_load import *

from deep_generalizability.postprocessing.sharpness_measures import *
from deep_generalizability.postprocessing.stats_plotting import *

from deep_generalizability.data_getters import *



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
    

def get_margins_filters(models, data, device=None, softmax_outputs=False):
    margins_filters = {}
    if device is not None:
        is_gpu = True
    else:
        is_gpu = False

    for k, m in models.items():
        
        inputs, labels = data
        if device is not None:
                inputs, labels = inputs.to(device).type(torch.cuda.FloatTensor), labels.to(device).type(
                    torch.cuda.LongTensor)

        outputs = get_model_outputs(m, data, softmax_outputs, device)

        _, predicted = torch.max(outputs, 1)

        correct_filter = predicted == labels
        
        # if correctly predicted
        second_largest = torch.topk(outputs, k=2, dim=1)
        

        curr_margins = second_largest[0][:, 0] - second_largest[0][:, 1]
        
        # we override the ones which were incorrectly predicted
        curr_margins[~correct_filter] = second_largest[0][:, 0][~correct_filter] - torch.Tensor(take_slice(outputs, labels))[~correct_filter]
        
        
        curr_margins = curr_margins.detach().numpy()
        correct_filter = correct_filter.detach().numpy()
        
        margins_filters[k] = (curr_margins, correct_filter)
    return margins_filters
    







if __name__ == "__main__":
    pass
