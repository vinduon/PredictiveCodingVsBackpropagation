"""
*    Title: On the relationship between predictive coding and backpropagation source code
*    Author: Robert Rosenbaum
*    Date: 2021
*    Code version: 1.0
*    Availability: https://github.com/RobertRosenbaum/PredictiveCodingVsBackProp
"""

import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


def load_dataset(train_batch_size, test_batch_size):
    train_dataset = MNIST('./', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = MNIST('./', train=False, transform=transforms.ToTensor(), download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=train_batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=test_batch_size,
                                              shuffle=True)

    return train_loader, test_loader
