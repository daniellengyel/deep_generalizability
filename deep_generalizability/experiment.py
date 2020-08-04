import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from ray import tune

from .utils import *
from .training import *
from .postprocessing import *
from .data_getters import get_data

import sys, os
import pickle

config = {}

# setting hyperparameters

# data specific
data_name = "concentric_balls"

if data_name == "CIFAR10":
    num_channels = 3
    height = 32
    width = height
    out_dim = 10
    inp_dim = height * width * num_channels
    config["data_meta"] = None
elif (data_name == "MNIST") or (data_name == "FashionMNIST"):
    num_channels = 1
    height = 28
    width = height
    out_dim = 10
    inp_dim = height * width * num_channels
    config["data_meta"] = None
elif data_name == "gaussian":
    inp_dim = 2
    out_dim = 2
    config["data_meta"] = None
elif data_name == "mis_gauss":
    inp_dim = 2
    out_dim = 2
    config["data_meta"] = None
elif data_name == "concentric_balls":
    inp_dim = 2
    out_dim = 2
    config["data_meta"] = {"inner_range": (0, 1), 
                           "outer_range": (2, 3),
                           "train_num_points_inner": 500,
                           "train_num_points_outer": 500,
                           "test_num_points_inner": 100,
                           "test_num_points_outer": 100,
                            }


config["data_name"] = data_name

# net
config["net_name"] = "SimpleNet"

if config["net_name"] == "SimpleNet":
    width = 256 # tune.grid_search([64])
    num_layers = 2 #  tune.grid_search([16])
    config["net_params"] = [inp_dim, out_dim, width, num_layers]
elif config["net_name"] == "LeNet":
    config["net_params"] = [height, width, num_channels, out_dim]

config["seed"] = 0

config["num_steps"] = 5000  # tune.grid_search([25000]) # roughly 50 * 500 / 16
config["mean_loss_threshold"] = None # 0.15

config["batch_train_size"] = tune.grid_search([32, 256, 4096])
config["batch_test_size"] = 16 # tune.grid_search([16])

config["var_noise"] = None # tune.grid_search(list(np.logspace(-4, -1, 5)))

config["ess_threshold"] = None  # tune.grid_search([0.97])
config["sampling_tau"] = 1 # tune.grid_search([1, 5]) # tune.grid_search([1, 25, 100])
config["sampling_wait"] = 0
config["sampling_stop"] = None
config["weight_type"] = "loss_gradient_weights"  # "input_output_forbenius", #

config["learning_rate"] = 0.1 # tune.grid_search(list(np.logspace(-3, 0, 10))) 
# config["lr_decay"] =
config["momentum"] = 0

config["num_nets"] = 2  # would like to make it like other one, where we can define region to initialize

config["softmax_beta"] = 0 # tune.grid_search([-50, 50]) #tune.grid_search([0] + list(-1*np.linspace(1, 100, 5)) + list(np.linspace(1, 100, 5))) # e.g. negtive to prioritize low weights
# offset = tune.grid_search([0.5, 0.25, 0.1])
config["softmax_adaptive"] = None  # [offset, 1000] # offset, and strength


config["device"] = "cpu"

config["hard_train_eps"] = None #  0.1

if data_name == "MNIST":
    config["reduce_train_per"] = 0.1
else:
    config["reduce_train_per"] = 1

# --- Set up folder in which to store all results ---
folder_name = get_file_stamp()
cwd = os.environ["PATH_TO_GEN_FOLDER"]
folder_path = os.path.join(cwd, "experiments", data_name, folder_name)
print(folder_path)
os.makedirs(folder_path)

# --- get data ---
train_data, test_data = get_data(data_name, vectorized=config["net_name"] == "SimpleNet",
                                 reduce_train_per=config["reduce_train_per"], seed=config["seed"], meta=config["data_meta"])

a = time.time()
# train(config, folder_path, train_data, test_data)
if config["device"] == "gpu":
    tune.run(lambda config_inp: train(config_inp, folder_path, train_data, test_data), config=config, resources_per_trial={'gpu': 1})
else:
    tune.run(lambda config_inp: train(config_inp, folder_path, train_data, test_data), config=config)

print(time.time() - a)


# TODO have logging of what we want to achieve with the current experiment.
# add a new distance metric. Distance from permutations -- used to know how many networks are needed.
# could also tell us how far away the vallys are and how symmetric the space is.
# Repeat experiments in finding minima with SGD paper
