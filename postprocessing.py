import numpy as np
import pandas as pd

import margin_flatness.postprocessing.postprocess_experiment as mf_post

import yaml, os, sys, re

import torch

import pickle


def main():
    # # # save analysis processsing

    root_folder = os.environ["PATH_TO_DEEP_FOLDER"]
    data_name = "MNIST"
    exp = "lr_MSE_test"
    experiment_folder = os.path.join(root_folder, "experiments", data_name, exp)

    # init torch
    is_gpu = False
    if is_gpu:
        torch.backends.cudnn.enabled = True
        device = torch.device("cuda:0")
    else:
        device = None
        # device = torch.device("cpu")

    # get_runs(experiment_folder, ["Loss", "Kish", "Potential", "Accuracy", "WeightVarTrace", "Norm",
    #                          "Trace", "Gradient"])  # TODO does not find acc and var

    
    # print("Getting Point Traces.")
    mf_post.get_exp_point_traces(experiment_folder, step=-1, seed=0, device=device, num_datapoints=100, on_test_set=False, should_cache=True)
    
    # get_exp_inp_out_jacobian(experiment_folder, step=-1, seed=0, device=device, num_datapoints=10, on_test_set=False, should_cache=True)


    # compute all point traces over time
    # f = lambda step: mf_post.get_exp_point_traces(experiment_folder, step=step, seed=0, device=device, num_datapoints=100, on_test_set=False, should_cache=True)
    # mf_post.get_all_steps_f(experiment_folder, f)

    # compute all loss over time 
    # f = lambda step: get_exp_loss_acc(experiment_folder, step, train_datapoints=1000, test_datapoints=1000, device=device)
    # get_all_steps_f(experiment_folder, f)


    # print("Getting Point Density.")
    # get_exp_point_eig_density(experiment_folder, -1, 0, device, num_datapoints=1000, on_test_set=False, should_cache=True)

    # get_exp_final_distances(experiment_folder, device=device)

    # get_exp_eig(experiment_folder, -1, num_eigenthings=5, FCN=True, device=device)
    # get_exp_trace(experiment_folder, -1, device=device)

    # mf_post.get_exp_loss_acc(experiment_folder, -1, train_datapoints=-1, test_datapoints=-1, device=device)

    # get_grad(experiment_folder, -1, False, FCN=True)

    # get_dirichlet_energy(experiment_folder, -1, num_steps=20, step_size=0.001, var_noise=0.5, alpha=1, seed=1, FCN=True)
    # get_exp_tsne(experiment_folder, -1)


if __name__ == "__main__":
    main()
    # import argparse
    #
    # parser = argparse.ArgumentParser(description='Postprocess experiment.')
    # parser.add_argument('exp_name', metavar='exp_name', type=str,
    #                     help='name of experiment')
    #
    # args = parser.parse_args()
    #
    # print(args)
    #
    # experiment_name = args.exp_name