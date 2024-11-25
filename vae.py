import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class __Encoder(nn.Module):
    def __init__(self, latent_space_dimension, *, stride=2):
        self.latent_space_dimension = latent_space_dimension
        self.relu_conv1 = nn.Conv2d(3, 32, 4, stride=stride)
        self.relu_conv2 = nn.Conv2d(32, 64, 4, stride=stride)
        self.relu_conv3 = nn.Conv2d(64, 128, 4, stride=stride)
        self.relu_conv4 = nn.Conv2d(128, 256, 4, stride=stride)

        self.dense_mu = nn.Linear(2 * 2 * 256, latent_space_dimension)
        self.dense_sigma = nn.Linear(2 * 2 * 256, latent_space_dimension)

    def forward(self, x):
        x = F.relu(self.relu_conv1(x))
        x = F.relu(self.relu_conv2(x))
        x = F.relu(self.relu_conv3(x))
        x = F.relu(self.relu_conv4(x))
        # Flatten the tensor for dense layers
        x = x.view(x.size(0), -1)
        mu = self.dense_mu(x)
        sigma = self.dense_sigma(x)
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        return z, mu, sigma


class __Decoder(nn.Module): ...


class ConvVAE(nn.Module):
    def __init__(
        self,
        latent_space_dimension=32,
    ):
        super().__init__()
        self.latent_space_dimension = latent_space_dimension
        # https://worldmodels.github.io/#:~:text=each%20convolution%20and%20deconvolution%20layer%20uses%20a%20stride%20of%202.
        self.encoder = __Encoder(latent_space_dimension, stride=2)


# if __name__ == "__main__":
#     import gymnasium as gym

#     env = gym.make("CarRacing-v2", continuous=False)
#     n_episodes = 10000
#     rewards = []
#     for episode in range(n_episodes):
#         total_reward = 0
#         done = False
#         s, _ = env.reset()
#         while not done:
#             action = env.action_space.sample()
#             s, reward, terminated, truncated, info = env.step(action)
#             print(s.shape)
#             done = terminated or truncated
#             total_reward += reward

#         rewards.append(total_reward)
#     print("Hello")
