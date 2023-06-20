"""
*    Title: On the relationship between predictive coding and backpropagation source code
*    Author: Robert Rosenbaum
*    Date: 2021
*    Code version: 1.0
*    Availability: https://github.com/RobertRosenbaum/PredictiveCodingVsBackProp
"""

import torch.nn as nn


def build_original_model():
    model = nn.Sequential(

        nn.Sequential(nn.Conv2d(1, 10, 3),
                      nn.ReLU(),
                      nn.MaxPool2d(2)
                      ),

        nn.Sequential(
            nn.Conv2d(10, 5, 3),
            nn.ReLU(),
            nn.Flatten()
        ),

        nn.Sequential(
            nn.Linear(5 * 11 * 11, 50),
            nn.ReLU()
        ),

        nn.Sequential(
            nn.Linear(50, 30),
            nn.ReLU()
        ),

        nn.Sequential(
            nn.Linear(30, 10)
        )

    )
    return model


def build_modified_model():
    model = nn.Sequential(
        # Layer 0 (Input): 28 x 28
        nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ),
        # Layer 1: (5x) 13 x 13
        nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        ),
        # Layer 2: (10x) 6 x 6
        nn.Sequential(
            nn.Linear(in_features=10 * 6 * 6, out_features=60),
            nn.BatchNorm1d(60),
            nn.ReLU()
        ),
        # Layer 3: Linear 60
        nn.Sequential(
            nn.Linear(in_features=60, out_features=10)
        )
        # Layer 4 (Output): Linear 10
    )
    return model
