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

config["seed"] = 0
config["device"] = "cpu"

# data specific
data_name = "CIFAR10"

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
    config["data_meta"] = {"dim": inp_dim,
                            "training_nums": 500,
                            "test_nums": 100}
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
config["reduce_train_per"] = 1

# net
config["net_name"] = "BatchNormSimpleNet"

if config["net_name"] == "SimpleNet":
    width = 256 # tune.grid_search([64])
    num_layers = 4 #  tune.grid_search([16])
    config["net_params"] = [inp_dim, out_dim, width, num_layers]
elif config["net_name"] == "LinearNet":
    config["net_params"] = [inp_dim, out_dim]
elif config["net_name"] == "BatchNormSimpleNet":
    config["net_params"] = [inp_dim, out_dim]
elif config["net_name"] == "LeNet":
    config["net_params"] = [height, width, num_channels, out_dim]

config["num_nets"] = 1  # would like to make it like other one, where we can define region to initialize

config["optimizer"] = "SGD" # "Adam"
config["learning_rate"] = tune.grid_search(list(np.logspace(-4, 0, 10))) 
config["momentum"] = 0

config["batch_train_size"] = 256 # tune.grid_search([16, 32, 256])
config["batch_test_size"] = 16 # tune.grid_search([16])

config["criterion"] = "cross-entropy" # "cross-entropy"

config["num_steps"] = 10000  # tune.grid_search([25000]) # roughly 50 * 500 / 16
config["mean_loss_threshold"] = None # 0.01 # 0.15


config["var_noise"] = None # tune.grid_search(list(np.logspace(-4, -1, 5)))

config["save_model_freq"] = 2500
config["print_stat_freq"] = 100


# --- Set up folder in which to store all results ---
folder_name = get_file_stamp()
cwd = os.environ["PATH_TO_DEEP_FOLDER"]
folder_path = os.path.join(cwd, "experiments", data_name, folder_name)
print(folder_path)
os.makedirs(folder_path)

# --- get data ---
train_data, test_data = get_data(data_name, vectorized=config["net_name"] in ["SimpleNet", "LinearNet", "BatchNormSimpleNet"],
                                 reduce_train_per=config["reduce_train_per"], seed=config["seed"], meta=config["data_meta"])

if config["device"] == "gpu":
    tune.run(lambda config_inp: train(config_inp, folder_path, train_data, test_data), config=config, resources_per_trial={'gpu': 1})
else:
    tune.run(lambda config_inp: train(config_inp, folder_path, train_data, test_data), config=config)

