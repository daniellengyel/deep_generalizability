import numpy as np
import matplotlib.pyplot as plt
import copy, yaml, pickle
import torch
import torchvision
import torch.optim as optim

from pyhessian import hessian

from torch.utils.tensorboard import SummaryWriter

from utils import *
from nets.Nets import SimpleNet, LeNet

from torch.utils.data import DataLoader

import os, time


def simple_training_step(net, iter_data_loader, optimizer, criterion, device=None):
    """Does update step on all networks and computes the weights.
    If wanting to do a random walk, set learning rate of net_optimizer to zero and set var_noise to noise level."""
    taking_step = True

    # get the inputs; data is a list of [inputs, labels]
    try:
        data = next(iter_data_loader)
    except:
        taking_step = False
        return net, taking_step, None

    inputs, labels = data
    if device is not None:
        inputs, labels = inputs.to(device).type(torch.cuda.FloatTensor), labels.to(device).type(
            torch.cuda.LongTensor)

    # Compute gradients for input.
    inputs.requires_grad = True

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward(retain_graph=True)
    optimizer.step()

    assert taking_step or (idx_net == 0)

    return net, taking_step, float(loss)
