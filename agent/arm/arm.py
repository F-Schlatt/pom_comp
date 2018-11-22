import copy
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset


class Arm():

    def __init__(self, path):

        self.v_network = Net(8, 3)
        self.q_network = Net(8, 9)

        self.load(path)

    def choose_action(self, images, scalar, actions):
        expected_value = self.v_network(images, scalar).detach().squeeze()
        cf_values = [self.q_network(images, scalar, action).detach().squeeze()
                     for action in actions]
        action_values = [max(torch.zeros(1), (cf_value - expected_value))
                         for cf_value in cf_values]
        action_probs = torch.tensor(action_values)
        if torch.sum(action_probs):
            action_probs = action_probs / torch.sum(action_probs)
        else:
            action_probs = torch.ones(len(actions)) / len(actions)
        action = int(torch.multinomial(action_probs, 1))
        return action

    def load(self, path):
        with open(path, 'rb') as fp:
            model = pickle.load(fp)

        self.v_network.load_state_dict(model['v_network'])
        self.q_network.load_state_dict(model['q_network'])


class Net(nn.Module):
    def __init__(self, img_input_dim, scalar_input_dim):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(img_input_dim, 10, 3)
        self.conv2 = nn.Conv2d(10, 10, 2)
        self.fc_in = nn.Linear(scalar_input_dim, 120)
        self.fc1 = nn.Linear(360, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 1)

    def forward(self, img, scalar):
        out = F.relu(self.conv1(img))
        out = F.relu(self.conv2(out))
        out = out.view(-1, self.num_flat_features(out))
        out = F.relu(self.fc1(out) + self.fc_in(scalar))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features