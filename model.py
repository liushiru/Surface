from __future__ import print_function, division

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.resnet_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)

        self.fc_count = 3
        self.fc_list = []
        self.dropout_list = []
        for i in range(self.fc_count):
            self.fc_list.append(nn.Linear(1000, 1000))
        for i in range(self.fc_count):
            self.dropout_list.append(nn.Dropout(0.2))
        self.fc_list = nn.ModuleList(self.fc_list)
        self.dropout_list = nn.ModuleList(self.dropout_list)
        self.out_layer = nn.Linear(1000, 100)

    def forward(self, x):
        # Pass data through conv1
        x = self.resnet_model(x)
        for layer in self.fc_list:
            x = torch.relu(layer(x))
        for layer in self.dropout_list:
            x = layer(x)
        output = self.out_layer(x)

        return output


