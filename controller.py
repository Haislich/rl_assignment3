import torch
from torch import nn
from vision import ConvVAE
from cmaes import cma.


class Controller(nn.Module):
    def __init__(self, latent_dimension=32, hidden_units=256, action_dimension=1):
        super().__init__()
        self.fc = nn.Linear(latent_dimension + hidden_units, action_dimension)

    def forward(self, latent: torch.Tensor, hidden: torch.Tensor):
        return torch.tanh(self.fc(torch.cat((latent, hidden), dim=1)))


class ControllerTrainer:
    def _rollout(controller, vision, memory):
        obs = env.reset()
        h = rnn.initial_state()
        done = False
        cumulative_reward = 0
        while not done:
            z = vae.encode(obs)
            a = controller.action([z, h])
            obs, reward, done = env.step(a)
            cumulative_reward += reward
            h = rnn.forward([a, z, h])
        return cumulative_reward
