import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Policy(nn.Module):
    continuous = False # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device

    def forward(self, x):
        # TODO
        return x
    
    def act(self, state):
        # TODO
        return 

    def train(self):
        # TODO
        return

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
