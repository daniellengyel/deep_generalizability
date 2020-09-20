########################################################################
# 2. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
import torch


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
   

    def __init__(self, inp_dim, out_dim, width, num_layers):
        super(SimpleNet, self).__init__()

        self.fc_input = nn.Linear(inp_dim, width)
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(num_layers - 1)])
        self.fc_final = nn.Linear(width, out_dim)

    def forward(self, x):
        x = F.relu(self.fc_input(x))
        for i in range(len(self.layers)):
            x = F.relu(self.layers[i](x))
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
    def __init__(self, inp_dim, out_dim):
        super(BatchNormSimpleNet, self).__init__()

        width = 256

        self.fc1 = nn.Linear(inp_dim, width)
        self.bn1 = nn.BatchNorm1d(num_features=width)
        self.fc2 = nn.Linear(width, width)
        self.bn2 = nn.BatchNorm1d(num_features=width)
        self.fc3 = nn.Linear(width, width)
        self.bn3 = nn.BatchNorm1d(num_features=width)
        # self.fc4 = nn.Linear(width, width)
        # self.bn4 = nn.BatchNorm1d(num_features=width)
        # self.fc5 = nn.Linear(width, width)
        # self.bn5 = nn.BatchNorm1d(num_features=width)

        self.fc_final = nn.Linear(width, out_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        # x = F.relu(self.bn4(self.fc4(x)))
        # x = F.relu(self.bn5(self.fc5(x)))
        x = self.fc_final(x)
        return x

class BatchNormSimpleNet(Module):
    def __init__(self, inp_dim, out_dim):
        super(BatchNormSimpleNet, self).__init__()

        width = 256

        self.fc1 = nn.Linear(inp_dim, width)
        self.bn1 = nn.BatchNorm1d(num_features=width)
        self.fc2 = nn.Linear(width, width)
        self.bn2 = nn.BatchNorm1d(num_features=width)
        self.fc3 = nn.Linear(width, width)
        self.bn3 = nn.BatchNorm1d(num_features=width)
        # self.fc4 = nn.Linear(width, width)
        # self.bn4 = nn.BatchNorm1d(num_features=width)
        # self.fc5 = nn.Linear(width, width)
        # self.bn5 = nn.BatchNorm1d(num_features=width)

        self.fc_final = nn.Linear(width, out_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        # x = F.relu(self.bn4(self.fc4(x)))
        # x = F.relu(self.bn5(self.fc5(x)))
        x = self.fc_final(x)
        return x

class KeskarC3(Module):
    def __init__(self, img_size, nb_classes):
        super(KeskarC3, self).__init__()

        model = Sequential()
        model.add(Convolution2D(64, 5, 5, border_mode='same', input_shape=img_size))
        self.conv1 = nn.Conv2d(out_channels=5, kernel_size=[5, 5])
        self.bn1 = nn.BatchNorm1d(num_features=width)
        model.add(BatchNormalization(mode=2,axis=1))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode='same'))
        self.pool1 = nn.MaxPool2d(2)


        model.add(Convolution2D(64, 5, 5, border_mode='same'))
        model.add(BatchNormalization(mode=2,axis=1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode='same'))

        model.add(Flatten())
        model.add(Dense(384))
        model.add(BatchNormalization(mode=2))
        model.add(Activation('relu'))
        model.add(Dropout(0.5)) 
        model.add(Dense(192))
        model.add(BatchNormalization(mode=2))
        model.add(Activation('relu'))
        model.add(Dropout(0.5)) 
        model.add(Dense(nb_classes, activation='softmax'))
        return model
        self.fc1 = nn.Linear(inp_dim, width)
        self.bn1 = nn.BatchNorm1d(num_features=width)
        self.fc2 = nn.Linear(width, width)
        self.bn2 = nn.BatchNorm1d(num_features=width)
        self.fc3 = nn.Linear(width, width)
        self.bn3 = nn.BatchNorm1d(num_features=width)
        # self.fc4 = nn.Linear(width, width)
        # self.bn4 = nn.BatchNorm1d(num_features=width)
        # self.fc5 = nn.Linear(width, width)
        # self.bn5 = nn.BatchNorm1d(num_features=width)

        self.fc_final = nn.Linear(width, out_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        # x = F.relu(self.bn4(self.fc4(x)))
        # x = F.relu(self.bn5(self.fc5(x)))
        x = self.fc_final(x)
        return x