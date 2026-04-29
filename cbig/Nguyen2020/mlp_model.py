#!/usr/bin/env python
"""MLP model for regression of 145 brain ROIs from 151-dim embeddings."""
import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    """
    Multi-layer perceptron for ROI regression.

    Maps a fixed-size embedding (X) to a vector of ROI volumes (Y).
    Architecture: Linear -> BN -> ReLU -> Dropout, repeated per hidden layer,
    followed by a linear output layer.
    """

    def __init__(self, input_size, output_size, hidden_sizes, dropout=0.0):
        """
        Args:
            input_size:   dimensionality of input embeddings (e.g. 151)
            output_size:  number of ROI targets to predict (e.g. 145)
            hidden_sizes: list of hidden layer widths, e.g. [512, 256]
            dropout:      dropout probability applied after each hidden layer
        """
        super(MLPRegressor, self).__init__()
        layers = []
        in_dim = input_size
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
