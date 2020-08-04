import numpy as np
import matplotlib.pyplot as plt
import copy, yaml, pickle
import torch
import torchvision
import torch.optim as optim

from .pyhessian import hessian

from torch.utils.tensorboard import SummaryWriter

from .utils import *
from .training_utils import *
from .nets.Nets import SimpleNet, LeNet
from .save_load import *

from torch.utils.data import DataLoader

import os, time


def train(config, folder_path, train_data, test_data):
    # init torch
    # TODO should also be a method. Probably in utils
    if config["device"] == "gpu":
        torch.backends.cudnn.enabled = True
        device = torch.device("cuda:0")
    else:
        device = None
        # device = torch.device("cpu")

    set_seed(config["seed"])

    # get dataloaders
    train_loader = DataLoader(train_data, batch_size=config["batch_train_size"], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config["batch_test_size"], shuffle=True)

    # Init neural nets and weights
    nets = get_nets(config["net_name"], config["net_params"], config["num_nets"], device=device)

    num_nets = config["num_nets"]
    nets_weights = np.zeros(num_nets)

    #  Define a Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    get_opt_func = get_optimizers(config)
    optimizers = get_opt_func(nets)

    # define stopping criterion
    stopping_criterion = get_stopping_criterion(config["num_steps"], config["mean_loss_threshold"])


    # init saving
    file_stamp = str(time.time())  # get_file_stamp()
    writer = SummaryWriter("{}/runs/{}".format(folder_path, file_stamp))
    with open("{}/runs/{}/{}".format(folder_path, file_stamp, "config.yml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    save_models(nets, config["net_name"], config["net_params"], folder_path, file_stamp, step=0)

    # init number of steps
    curr_step = 1
    mean_loss = float("inf")

    # train
    while (not stopping_criterion(mean_loss, curr_step)):

        # get train loaders for each net
        net_data_loaders = [iter(train_loader) for _ in range(num_nets)]

        is_training_curr = True
        while is_training_curr and (not stopping_criterion(mean_loss, curr_step)):
            if (curr_step % 100) == 1:
                print("Step: {}".format(curr_step))
                print("Mean Loss: {}".format(mean_loss))

            # do update step for each net
            nets, nets_weights, steps_taken, mean_loss_after_step = training_step(nets, optimizers,
                                                                                net_data_loaders, criterion,
                                                                                weight_type, var_noise=config["var_noise"], curr_step=curr_step,
                                                                                writer=writer, device=device)
            
            if steps_taken == 0:
                break
            else:
                mean_loss = mean_loss_after_step

            # # Get variation of network weights
            covs = get_params_cov(nets)
            writer.add_scalar('WeightVarTrace/', torch.norm(covs), curr_step)

        

            # update curr_step
            curr_step += steps_taken

        # # get test error
        # for idx_net in range(num_nets):
        #     accuracy = get_net_accuracy(nets[idx_net], test_loader, device=device)
        #     writer.add_scalar('Accuracy/net_{}'.format(idx_net), accuracy, curr_step)

    # save final nets
    save_models(nets, config["net_name"], config["net_params"], folder_path, file_stamp, step=curr_step)

    return nets

def training_step(nets, net_optimizers, net_data_loaders, criterion, weight_type, var_noise=None,
                   curr_step=0, writer=None, device=None):
    """Does update step on all networks and computes the weights.
    If wanting to do a random walk, set learning rate of net_optimizer to zero and set var_noise to noise level."""
    taking_step = True

    mean_loss = 0

    for idx_net in range(len(nets)):

        # get net and optimizer
        net = nets[idx_net]
        optimizer = net_optimizers[idx_net]

        # get the inputs; data is a list of [inputs, labels]
        try:
            data = next(net_data_loaders[idx_net])
            print(data)
        except:
            taking_step = False
            break
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

        if var_noise is not None:
            net = add_noise(net, var_noise, device)

        # get gradient
        param_grads = get_grad_params_vec(net)
        curr_grad = torch.norm(param_grads)


        # store metrics for each net
        if writer is not None:
            writer.add_scalar('Loss/train/net_{}'.format(idx_net), loss, curr_step)
            writer.add_scalar('Gradient/train/net_{}'.format(idx_net), curr_grad, curr_step)
            writer.add_scalar('Norm/net_{}'.format(idx_net), torch.norm(get_params_vec(net)), curr_step)
            # if (curr_step % 50) == 0:
            #     # a = time.time()
            #     is_gpu = device is not None
            #     trace = np.mean(hessian(net, criterion, data=(inputs, labels), cuda=is_gpu).trace())
            #     writer.add_scalar('Trace/net_{}'.format(idx_net), trace, curr_step)
            #     # print("Getting trace took {}".format(time.time() - a))

        mean_loss += float(loss)

    assert taking_step or (idx_net == 0)

    return nets, nets_weights, 1*taking_step, mean_loss / len(nets)
