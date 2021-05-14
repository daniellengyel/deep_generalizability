import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from ray import tune
import ray

import margin_flatness as mf

import sys, os
import pickle

config = {}

# Job specific 
ARRAY_INDEX = int(os.environ["PBS_ARRAY_INDEX"]) - 1

# data specific
data_name = "KMNIST"

if data_name == "CIFAR10":
    num_channels = 3
    height = 32
    width = height
    out_dim = 10
    inp_dim = height * width * num_channels
    config["data_meta"] = None
elif (data_name == "MNIST") or (data_name == "FashionMNIST") or (data_name == "KMNIST"):
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
config["net_name"] = "SimpleNet"
if ARRAY_INDEX // 3 == 0:
    activation_function = "relu"
elif ARRAY_INDEX // 3 == 1:
    activation_function = "sigmoid"
elif ARRAY_INDEX // 3 == 2:
    activation_function = "tanh" 


if config["net_name"] == "SimpleNet":
    width = 256 
    num_layers = 4 
    dropout_p = 0

    config["net_params"] = [inp_dim, out_dim, width, num_layers, dropout_p, activation_function]
elif config["net_name"] == "LinearNet":
    config["net_params"] = [inp_dim, out_dim]
elif config["net_name"] == "BatchNormSimpleNet":
    width = 256 # tune.grid_search([32, 64, 256, 512])
    config["net_params"] = [inp_dim, out_dim, width, activation_function]
elif config["net_name"] == "LeNet":
    config["net_params"] = [height, width, num_channels, out_dim]
elif config["net_name"] == "KeskarC3":
    config["net_params"] = [height, width, num_channels, out_dim]

config["num_nets"] = 1  # would like to make it like other one, where we can define region to initialize

config["optimizer"] = "SGD" # tune.grid_search(["SGD"])

if ARRAY_INDEX % 3 == 0:
    config["optimizer"] = tune.grid_search(["SGD"])
elif ARRAY_INDEX % 3 == 1:
    config["optimizer"] = tune.grid_search(["Adam"])
elif ARRAY_INDEX % 3 == 2:
    config["optimizer"] = tune.grid_search(["RMSProp"])

config["weight_decay"] = tune.grid_search([0, 0.0001, 0.0005])  # l2 penalty 
config["learning_rate"] = tune.grid_search([0.1, 0.05, 0.01, 0.005, 0.001]) # tune.grid_search([1, 0.25, 0.1, 0.05, 0.01, 0.001])
config["momentum"] = 0.9
config["learning_rate_schedule"] = None # {"name": "step", "gamma": 0.75, "step_size": 10000} # step size is number of steps until applying multiplicative gamma

config["batch_train_size"] = tune.grid_search([32, 64, 256])
config["batch_test_size"] = 1 # tune.grid_search([16])

config["criterion"] = "MSE"

config["num_steps"] = 25000  # tune.grid_search([25000]) # roughly 50 * 500 / 16
config["mean_loss_threshold"] = 0 # 0.0005 # 0.01 # 0.15


config["save_model_freq"] = 10000
config["print_stat_freq"] = 1000

config["seed"] = tune.grid_search([0, 5, 10])
config["device"] = "cpu"
config["data_seed"] = 0 # should generally not be changed. 


# --- Set up folder in which to store all results ---
folder_name = mf.utils.get_file_stamp()
cwd = os.environ["PATH_TO_DEEP_FOLDER"]
folder_path = os.path.join(cwd, "experiments", data_name, folder_name)
print(folder_path)
os.makedirs(folder_path)

# --- get data ---
train_data, test_data = mf.data_getters.get_data(data_name, vectorized=config["net_name"] in ["SimpleNet", "LinearNet", "BatchNormSimpleNet"],
                                 reduce_train_per=config["reduce_train_per"], seed=config["data_seed"], meta=config["data_meta"])


# ray.shutdown()
ray.init(_temp_dir='/rds/general/user/dl2119/ephemeral', num_cpus=32)

if config["device"] == "gpu":
    tune.run(lambda config_inp: mf.training.train(config_inp, folder_path, train_data, test_data), config=config, resources_per_trial={'gpu': 1})
else:
    tune.run(lambda config_inp: mf.training.train(config_inp, folder_path, train_data, test_data), config=config)

