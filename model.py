from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.resnet_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        self.fc1 = nn.Linear(1000, 100)
        self.fc2 = nn.Linear(100, 3)


    def forward(self, x):
        # Pass data through conv1
        x = self.resnet_model(x)
        x = self.fc1(x)
        output = self.fc2(x)

        return output

