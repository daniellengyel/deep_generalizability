import os
from margin_flatness.utils import get_time_stamp
import shutil, errno
import time

def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise

exp_path = "/rds/general/user/dl2119/home/deep_generalizability/experiments"
exp_data = "CIFAR10"
exp_path = os.path.join(exp_path, exp_data)
# exp_names = ["May17_16-27-02_cx1-103-9-1.cx1.hpc.ic.ac.uk", "May17_16-29-12_cx1-103-9-4.cx1.hpc.ic.ac.uk", "May17_16-40-06_cx3-2-5.cx3.hpc.ic.ac.uk", "May17_17-21-55_cx3-5-5.cx3.hpc.ic.ac.uk"]
exp_names =[
        "May23_17-15-55_cx1-103-16-3.cx1.hpc.ic.ac.uk",
        "May23_17-15-50_cx3-2-6.cx3.hpc.ic.ac.uk",
        "May23_17-16-11_cx1-105-13-4.cx1.hpc.ic.ac.uk",
        "May23_17-16-11_cx3-6-25.cx3.hpc.ic.ac.uk",
        "May23_17-41-18_cx3-6-17.cx3.hpc.ic.ac.uk",
        "May23_18-19-31_cx3-6-6.cx3.hpc.ic.ac.uk"
    ]
target_name = "LeNet_short"

target_folder = os.path.join(exp_path, target_name)
if not os.path.isdir(target_folder):
    os.makedirs(target_folder)

model_target_folder = os.path.join(target_folder, "models")
if not os.path.isdir(model_target_folder):
    os.makedirs(model_target_folder)

run_target_folder = os.path.join(target_folder, "runs")
if not os.path.isdir(run_target_folder):
    os.makedirs(run_target_folder)

postprocessing_target_folder = os.path.join(target_folder, "postprocessing")
if not os.path.isdir(postprocessing_target_folder):
    os.makedirs(postprocessing_target_folder)

curr_time_stamp = get_time_stamp()
for exp_name in exp_names:
    print(exp_name)
    print("models")
    for model_dir in os.listdir(os.path.join(exp_path, exp_name, "models")):
        copyanything(os.path.join(exp_path, exp_name, "models", model_dir), os.path.join(model_target_folder, model_dir))
    print("runs")
    for run_dir in os.listdir(os.path.join(exp_path, exp_name, "runs")):
        copyanything(os.path.join(exp_path, exp_name, "runs", run_dir), os.path.join(run_target_folder, run_dir))
    # print("post")
    # for post_type in os.listdir(os.path.join(exp_path, exp_name, "postprocessing")):
    #     if not os.path.isdir(os.path.join(postprocessing_target_folder, post_type, "step_-1")):
    #         os.makedirs(os.path.join(postprocessing_target_folder, post_type, "step_-1"))
    #     for time_stamp in os.listdir(os.path.join(exp_path, exp_name, "postprocessing", post_type, "step_-1")):
    #        for post_exp_dir in os.listdir(os.path.join(exp_path, exp_name, "postprocessing", post_type, "step_-1", time_stamp)):
    #            if not os.path.isdir(os.path.join(exp_path, exp_name, "postprocessing", post_type, "step_-1", time_stamp, post_exp_dir)):
    #                continue
    #            copyanything(os.path.join(exp_path, exp_name, "postprocessing", post_type, "step_-1", time_stamp, post_exp_dir), os.path.join(postprocessing_target_folder, post_type, "step_-1", curr_time_stamp, post_exp_dir))