import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class __Encoder(nn.Module):
    def __init__(self, latent_dimension, *, stride=2):
        super().__init__()
        self.relu_conv1 = nn.Conv2d(3, 32, 4, stride=stride)
        self.relu_conv2 = nn.Conv2d(32, 64, 4, stride=stride)
        self.relu_conv3 = nn.Conv2d(64, 128, 4, stride=stride)
        self.relu_conv4 = nn.Conv2d(128, 256, 4, stride=stride)

        self.fc_mu = nn.Linear(2 * 2 * 256, latent_dimension)
        self.fc_sigma = nn.Linear(2 * 2 * 256, latent_dimension)

    def forward(self, x):
        x = F.relu(self.relu_conv1(x))
        x = F.relu(self.relu_conv2(x))
        x = F.relu(self.relu_conv3(x))
        x = F.relu(self.relu_conv4(x))
        # Flatten the tensor for dense layers
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_sigma = self.fc_sigma(x)

        # This reparametrization trick allow us to still be able to train it
        return mu, log_sigma


class __Decoder(nn.Module):
    def __init__(self, latent_dimension, image_chanels, *, stride=2):
        super().__init__()
        self.fc = nn.Linear(latent_dimension, 1024)
        self.relu_deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=stride)
        self.relu_deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=stride)
        self.relu_deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=stride)
        self.sigmoid_deconv = nn.ConvTranspose2d(32, image_chanels, 6, stride=stride)

    def forward(self, x):
        x = self.fc(x)
        x = x.unsqueeze(-1).unsqueeze(-1)

        x = F.relu(self.relu_deconv1(x))
        x = F.relu(self.relu_deconv2(x))
        x = F.relu(self.relu_deconv3(x))
        # https://github.com/pytorch/pytorch/issues/65910
        return torch.sigmoid(self.sigmoid_deconv(x))


class ConvVAE(nn.Module):
    def __init__(self, latent_dimension=32, image_channels=3):
        super().__init__()
        self.latent_dimension = latent_dimension
        # https://worldmodels.github.io/#:~:text=each%20convolution%20and%20deconvolution%20layer%20uses%20a%20stride%20of%202.
        self.encoder = __Encoder(latent_dimension, stride=2)
        self.decoder = __Decoder(latent_dimension, image_channels, stride=2)

    def forward(self, x):
        mu, log_sigma = self.encoder(x)
        sigma = log_sigma.exp()
        z = mu + sigma * torch.randn_like(sigma)
        reconstruction = self.decoder(z)
        return reconstruction

    def get_latent(self, observation: torch.Tensor):
        mu, log_sigma = self.encoder(observation)
        sigma = log_sigma.exp()
        return mu + sigma * torch.randn_like(sigma)
