import dataset
import vision
import memory
import torch
from torch import nn
import controller
from torch.distributions import Categorical, Normal


def sample_from_gmm(pi, mu, sigma):

    batch_size, num_mixtures, latent_dim = mu.size()

    #  Sample mixture component indices based on pi
    categorical = Categorical(pi)
    component_indices = categorical.sample()

    # Select the corresponding mu and sigma
    selected_mu = mu[torch.arange(batch_size), component_indices, :]
    selected_sigma = sigma[torch.arange(batch_size), component_indices, :]

    # Sample from the selected Gaussian
    normal = Normal(selected_mu, selected_sigma)
    sampled_z = normal.sample()

    return sampled_z


class WorldModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.V = vision.ConvVAE()
        self.M = memory.MDN_RNN()
        self.C = controller.Controller()

    def forward(self, obs, action, hidden=None):

        # Step 1: Encode the observation into the latent space using the VAE
        z_t = self.V.encode(obs)  # (BATCH_SIZE, LATENT_DIM)

        # Step 2: Update the memory using the MDN-RNN
        # Input to MDN-RNN: latent vector z_t and action a_t
        _, _, _, hidden = self.M.forward(z_t, action, hidden)  # MDN-RNN output

        # Step 3: Use the Controller to generate the next action
        # Input to Controller: latent vector z_t and hidden state h_t
        next_action = self.C.forward(z_t, hidden)

        return next_action, hidden, z_t

    def train(self, dataset: dataset.RolloutDataset, temperature: int):
        # Train the VAE
        ...
