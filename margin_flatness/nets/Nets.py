import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
import torch
import numpy as np

import functools
import operator


class LeNet(Module):
    def __init__(self, height, width, channels, out_dim):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(int(16 * (height - 12)/4 * (width - 12)/4), 120) # Each conv subtracts by 4 and MaxPool divides size by 2.
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, out_dim)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y


class SimpleNet(Module):
    def __init__(self, inp_dim, out_dim, width, num_layers, dropout_p=0, activation=None):
        super(SimpleNet, self).__init__()

        self.dropout_p = dropout_p
        
        if activation is None or activation == "relu":
            self.activation = F.relu
        if activation == "sigmoid":
            self.activation = F.hardsigmoid
        if activation == "tanh":
            self.activation = F.hardtanh

        self.fc_input = nn.Linear(inp_dim, width)
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(num_layers - 1)])
        self.layer_dropouts = nn.ModuleList([nn.Dropout(p=self.dropout_p) for _ in range(num_layers - 1)])
        self.fc_final = nn.Linear(width, out_dim)

    def forward(self, x):
        x = self.activation(self.fc_input(x))
        for i in range(len(self.layers)):
            x = self.activation(self.layers[i](x))
            x = self.layer_dropouts[i](x)
        x = self.fc_final(x)
        return x

class LinearNet(Module):
    def __init__(self, inp_dim, out_dim):
        super(LinearNet, self).__init__()

        self.fc_layer = nn.Linear(inp_dim, out_dim)
    

    def forward(self, x):
        x = self.fc_layer(x)
        return x


class BatchNormSimpleNet(Module):
    def __init__(self, inp_dim, out_dim, width=None, activation=None):
        super(BatchNormSimpleNet, self).__init__()

        if width is None:
            width = 256

        if activation is None or activation == "relu":
            self.activation = F.relu
        if activation == "sigmoid":
            self.activation = F.hardsigmoid

        self.fc1 = nn.Linear(inp_dim, width)
        self.bn1 = nn.BatchNorm1d(num_features=width)
        self.fc2 = nn.Linear(width, width)
        self.bn2 = nn.BatchNorm1d(num_features=width)
        self.fc3 = nn.Linear(width, width)
        self.bn3 = nn.BatchNorm1d(num_features=width)
        self.fc4 = nn.Linear(width, width)
        self.bn4 = nn.BatchNorm1d(num_features=width)
        self.fc5 = nn.Linear(width, width)
        self.bn5 = nn.BatchNorm1d(num_features=width)

        self.fc_final = nn.Linear(width, out_dim)

    def forward(self, x):
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.activation(self.bn3(self.fc3(x)))
        x = self.activation(self.bn4(self.fc4(x)))
        x = self.activation(self.bn5(self.fc5(x)))
        x = self.fc_final(x)
        return x



    
class KeskarC3(Module):

    def __init__(self, height, width, channels, out_dim):
        super(KeskarC3, self).__init__()

        num_filters = 24

        H_conv_padding = int((height + 3)/2) + 1 - (height % 2)
        W_conv_padding = int((width + 3)/2) + 1 - (width % 2)

        self.H_pool_padding = int((height + 1)/2) + 1 - (height % 2)
        self.W_pool_padding = int((width + 1)/2) + 1 - (width % 2)

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=num_filters, kernel_size=[5, 5], stride=2, padding=(H_conv_padding, W_conv_padding))
        self.bn1 = nn.BatchNorm2d(num_features=num_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)

        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=[5, 5], stride=2, padding=(H_conv_padding, W_conv_padding))
        self.bn2 = nn.BatchNorm2d(num_features=num_filters)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)

        self.num_features_before_fcnn = num_filters*height*width

        self.fc1 = nn.Linear(in_features=self.num_features_before_fcnn, out_features=192)
        self.bn3 = nn.BatchNorm1d(num_features=192)
        self.dp1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(in_features=192, out_features=86)
        self.bn4 = nn.BatchNorm1d(num_features=86)
        self.dp2 = nn.Dropout(p=0.5)

        self.out_layer = nn.Linear(in_features=86, out_features=out_dim)
        

    def forward(self, x):
        batch_size = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.pad(x, pad=(self.H_pool_padding, self.H_pool_padding, self.W_pool_padding, self.W_pool_padding))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.pad(x, pad=(self.H_pool_padding, self.H_pool_padding, self.W_pool_padding, self.W_pool_padding))
        x = self.pool2(x)

        x = x.view(batch_size, self.num_features_before_fcnn)  # flatten the vector

        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dp1(x)

        x = F.relu(self.bn4(self.fc2(x)))
        x = self.dp2(x)

        x = self.out_layer(x)

        return x

class NormOutputNet(Module):

    def __init__(self, model):
        super(NormOutputNet, self).__init__()
        self.model = model
        self.norm_scale = None

    def forward(self, x):
        if self.norm_scale is None:
            self._set_norm_scale(x)

        y = self.model(x)
        # print(torch.norm(y[0]))
        y = y * 25#/ self.norm_scale
        return y
    
    def _set_norm_scale(self, x):
        unit_input = torch.ones(x[0].shape)[np.newaxis, :]
        print(torch.norm(self.model(x)))
        unit_input /= (torch.norm(unit_input))
        self.norm_scale =  torch.norm(self.model(unit_input)) / 10  # torch.norm(self.model(x)) / 25
        print(self.norm_scale)
# class UnitvectorOutputNet(Module):

#     def __init__(self, model):
#         super(UnitvectorOutputNet, self).__init__()
#         self.model = model

#     def forward(self, x):
#         y = self.model(x)
#         y = y / torch.norm(y, dim=1)
#         return y

# class ScaledOutputNet(Module):

#     def __init__(self, model, scale=20):
#         super(ScaledOutputNet, self).__init__()
#         self.model = model
#         self.scale = scale

#     def forward(self, x):
#         y = self.model(x)
#         y = y * self.scale
#         return y