


# Define the experiment folder r
root_folder = os.environ["PATH_TO_DEEP_FOLDER"]
data_name = "MNIST"
exp = "LeNet_mse"
experiment_folder = os.path.join(root_folder, "experiments", data_name, exp)

# init torch
is_gpu = False
if is_gpu:
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda:0")
else:
    device = None
    # device = torch.device("cpu")



# then really want to get the margin and tarces, i guess all i need is X_data, Y_data. Cool 