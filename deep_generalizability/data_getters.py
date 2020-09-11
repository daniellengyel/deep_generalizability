import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import os, pickle


from .save_load import *

from random import random



PATH_TO_DATA = "{}/data".format(os.environ["PATH_TO_DEEP_FOLDER"])

def get_data(data_name, vectorized=False, reduce_train_per=None, seed=0, meta=None):
    if data_name == "gaussian":
        train_data, test_data = _get_gaussian(meta=meta, seed=seed)
    elif data_name == "concentric_balls":
        train_data, test_data = _get_concentric_balls(meta=meta, seed=seed)
    elif data_name == "mis_gauss":
        train_data, test_data = _get_mis_gauss(seed=seed)
    elif data_name == "MNIST":
        train_data, test_data = _get_MNIST()
    elif data_name == "CIFAR10":
        train_data, test_data = _get_CIFAR10()
    elif data_name == "FashionMNIST":
        train_data, test_data = _get_FashionMNIST()
    elif data_name == "ImageNet":
        train_data, test_data = _get_ImageNet()
    else:
        raise NotImplementedError("{} is not implemented.".format(data_name))
    if reduce_train_per is not None:
        train_data = ReducedData(train_data, per=reduce_train_per, seed=seed)
    if vectorized:
        train_data, test_data = VectorizedWrapper(train_data), VectorizedWrapper(test_data)
    return train_data, test_data

# def get_random_data_subset(data, num_datapoints=1, seed=0):
#     return ReducedData(data, num_datapoints=num_datapoints, seed=seed)

def get_random_data_subset(data, num_datapoints=1, seed=0):
    set_seed(seed)
    data_loader = DataLoader(data, batch_size=num_datapoints, shuffle=True)
    return DataWrapper(next(iter(data_loader)))


def _get_MNIST():
    train_data = torchvision.datasets.MNIST(os.path.join(PATH_TO_DATA, "MNIST"), train=True,
                                            download=True,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                    (0.1307,), (0.3081,))
                                            ]))
    test_data = torchvision.datasets.MNIST(os.path.join(PATH_TO_DATA, "MNIST"), train=False,
                                           download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   (0.1307,), (0.3081,))
                                           ]))

    return train_data, test_data


def _get_FashionMNIST():
    train_data = torchvision.datasets.FashionMNIST(
        root=os.path.join(PATH_TO_DATA, "FashionMNIST"),
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    test_data = torchvision.datasets.FashionMNIST(
        root=os.path.join(PATH_TO_DATA, "FashionMNIST"),
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    return train_data, test_data

def _get_ImageNet():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    transform = transforms.Compose(
            [transforms.ToTensor(),
            normalize])

    train_data = torchvision.datasets.ImageNet(root=os.path.join(PATH_TO_DATA, "ImageNet"), train=True, 
                                            download=True, transform=transform)
    test_data = torchvision.datasets.ImageNet(root=os.path.join(PATH_TO_DATA, "ImageNet"), train=False,
                                            download=True, transform=transform)
    return train_data, test_data


def _get_CIFAR10():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = torchvision.datasets.CIFAR10(root=os.path.join(PATH_TO_DATA, "CIFAR10"), train=True,
                                            download=True, transform=transform)

    test_data = torchvision.datasets.CIFAR10(root=os.path.join(PATH_TO_DATA, "CIFAR10"), train=False,
                                           download=True, transform=transform)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_data, test_data

def _get_concentric_balls(meta, seed=0):
    set_seed(seed)
    dim = 2
    inner_range = meta["inner_range"]
    outer_range = meta["outer_range"]
    train_num_points_inner = meta["train_num_points_inner"]
    train_num_points_outer = meta["train_num_points_outer"]
    test_num_points_inner = meta["test_num_points_inner"]
    test_num_points_outer = meta["test_num_points_outer"]

    train_data = ConcentricSphere(dim, inner_range, outer_range, train_num_points_inner,
                 train_num_points_outer)
    test_data = ConcentricSphere(dim, inner_range, outer_range, test_num_points_inner,
                 test_num_points_outer)

    return train_data, test_data

def _get_gaussian(meta, seed=0):
    set_seed(seed)
    # get data
    gaussian_params = []

    dim = meta["dim"]

    means = np.eye(dim)
    covs = [np.eye(dim) for _ in range(dim)]
    
    training_nums = meta["training_nums"]
    test_nums = meta["test_nums"]

    train_gaussian = GaussianMixture(means, covs, len(means) * [training_nums])
    test_gaussian = GaussianMixture(means, covs, len(means) * [test_nums])

    return train_gaussian, test_gaussian

def _get_mis_gauss(seed=0):
    set_seed = seed
    # get data
    gaussian_params = []

    cov_1 = np.array([[1, 0], [0, 1]])

    mean_1 = np.array([0, 0])

    means = [mean_1]
    covs = [cov_1]
    training_nums = 200
    test_nums = 120000

    train_gaussian = MisGauss(means, covs, len(means) * [training_nums])
    test_gaussian = MisGauss(means, covs, len(means) * [test_nums])

    return train_gaussian, test_gaussian

class DataWrapper(Dataset):
    """Wrapper for data of the form:
    data[0]: Torch Input data, data[1]: Torch Output Data"""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[0][index], self.data[1][index]
    
    def __len__(self):
        return len(self.data[0])

class GaussianMixture(Dataset):
    """Dataset gaussian mixture. Points of first gaussian are mapped to 0 while points in the second are mapped 1.

    Parameters
    ----------
    means:
        i: mean
    covs:
        i: cov
    nums:
        i: num for ith class
    """

    def __init__(self, means, covs, nums):

        self.data = []
        self.targets = []

        self.num_classes = len(covs)

        xs = None
        ys = []
        for i in range(len(covs)):
            mean = means[i]
            cov = covs[i]
            num = nums[i]
            x = np.random.multivariate_normal(mean, cov, num)
            if xs is None:
                xs = x
            else:
                xs = np.concatenate([xs, x], axis=0)
            ys += num * [i]

        self.data = torch.Tensor(xs)

        targets = np.array(ys)  # np.eye(self.num_classes)[ys]
        self.targets = torch.Tensor(targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index].long()

    def __len__(self):
        return len(self.data)


class MisGauss(Dataset):
    """Dataset gaussian mixture. Points of first gaussian are mapped to 0 while points in the second are mapped 1.

    Parameters
    ----------
    means:
        i: mean
    covs:
        i: cov
    nums:
        i: num for ith class
    """

    def __init__(self, means, covs, nums):
        self.data = []
        self.targets = []

        self.num_classes = len(covs)

        xs = None
        ys = []
        for i in range(len(covs)):
            mean = means[i]
            cov = covs[i]
            num = nums[i]
            x = np.random.multivariate_normal(mean, cov, num)
            y = 1 * (x[:, 0] >= 0)
            y = [self._flip(a, 0.05) for a in y]
            x += np.random.normal(scale=0.15, size=x.shape)

            if xs is None:
                xs = x
            else:
                xs = np.concatenate([xs, x], axis=0)

            ys += y

        self.data = torch.Tensor(xs)

        targets = np.array(ys)  # np.eye(self.num_classes)[ys]
        self.targets = torch.Tensor(targets)

    def _flip(self, val, alpha=0.05):
        should_flip = np.random.uniform(0, 1) < alpha
        if should_flip:
            if val == 0:
                val = 1
            elif val == 1:
                val = 0
        return val

    def __getitem__(self, index):
        return self.data[index], self.targets[index].long()

    def __len__(self):
        return len(self.data)

#  vectorizes data
class VectorizedWrapper():
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        data, target = self.data.__getitem__(item)
        return data.view(-1), target

    def __len__(self):
        return len(self.data)

class ReducedData:
    def __init__(self, data, per=None, num_datapoints=None, seed=0):
        assert (per is None) or (num_datapoints is None)
        set_seed(seed)
        if per is not None:
            self.num_datapoints = int(len(data) * per)
        else:
            self.num_datapoints = num_datapoints
        self.idxs = np.random.choice(list(range(len(data))), self.num_datapoints)
        self.data = data

    def __getitem__(self, item):
        data, target = self.data.__getitem__(self.idxs[item])
        return data, target

    def __len__(self):
        return self.num_datapoints

class ConcentricSphere(Dataset):
    """Dataset of concentric d-dimensional spheres. Points in the inner sphere
    are mapped to -1, while points in the outer sphere are mapped 1.

    Parameters
    ----------
    dim : int
        Dimension of spheres.

    inner_range : (float, float)
        Minimum and maximum radius of inner sphere. For example if inner_range
        is (1., 2.) then all points in inner sphere will lie a distance of
        between 1.0 and 2.0 from the origin.

    outer_range : (float, float)
        Minimum and maximum radius of outer sphere.

    num_points_inner : int
        Number of points in inner cluster

    num_points_outer : int
        Number of points in outer cluster
    """
    def __init__(self, dim, inner_range, outer_range, num_points_inner,
                 num_points_outer):
        self.dim = dim
        self.inner_range = inner_range
        self.outer_range = outer_range
        self.num_points_inner = num_points_inner
        self.num_points_outer = num_points_outer

        self.data = []
        self.targets = []

        # Generate data for inner sphere
        for _ in range(self.num_points_inner):
            self.data.append(
                random_point_in_sphere(dim, inner_range[0], inner_range[1])
            )
            # self.targets.append(torch.Tensor([-1]))
            self.targets.append(0)


        # Generate data for outer sphere
        for _ in range(self.num_points_outer):
            self.data.append(
                random_point_in_sphere(dim, outer_range[0], outer_range[1])
            )
            self.targets.append(1)
        
        self.targets = torch.Tensor(self.targets)


    def __getitem__(self, index):
        return self.data[index], self.targets[index].long()

    def __len__(self):
        return len(self.data)

def random_point_in_sphere(dim, min_radius, max_radius):
    """Returns a point sampled uniformly at random from a sphere if min_radius
    is 0. Else samples a point approximately uniformly on annulus.

    Parameters
    ----------
    dim : int
        Dimension of sphere

    min_radius : float
        Minimum distance of sampled point from origin.

    max_radius : float
        Maximum distance of sampled point from origin.
    """
    # Sample distance of point from origin
    unif = random()
    distance = (max_radius - min_radius) * (unif ** (1. / dim)) + min_radius
    # Sample direction of point away from origin
    direction = torch.randn(dim)
    unit_direction = direction / torch.norm(direction, 2)
    return distance * unit_direction
