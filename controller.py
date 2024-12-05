import torch
from torch import nn
from vision import ConvVAE
from cma import CMA


class Controller(nn.Module):

    def __init__(self, latent_dimension=32, hidden_units=256, action_dimension=1):
        super().__init__()
        self.fc = nn.Linear(latent_dimension + hidden_units, action_dimension)

    def forward(self, latent: torch.Tensor, hidden: torch.Tensor):
        return torch.tanh(self.fc(torch.cat((latent, hidden), dim=1)))


class ControllerTrainer: ...
