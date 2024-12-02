from pathlib import Path
from numpy import mean
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import RolloutDataloader, RolloutDataset
from vision import ConvVAE


# Credits:
# - https://github.com/sksq96/pytorch-mdn/blob/master/mdn-rnn.ipynb
# - https://github.com/hardmaru/WorldModelsExperiments/blob/master/carracing/rnn/rnn.py
#
class MDN_RNN(nn.Module):
    def __init__(self, latent_dimension=32, hidden_units=256, num_mixtures=5):
        super().__init__()
        self.hidden_dim = hidden_units
        self.num_mixtures = num_mixtures
        self.action_embedding = nn.Embedding(6, latent_dimension)
        # The *2 is not taken from the original paper, the idea is the following.
        # We want to condition the latent on the actions, however while latents have 32 dimensions
        # actions are scalars, so actions are embedded in a latest_dimensional space,
        # This will result in actions and observation have an equal weight in the resulting LSTM.
        self.rnn = nn.LSTM(2 * latent_dimension, hidden_units, batch_first=True)
        # The MDN is trying to predict the probability distribution of the next latent vector.
        # Instead of predicting a single point value (like a regression model),
        # the MDN models the output as a Mixture of Gaussian Distributions (MoG).
        # This allows the model to handle multimodal data, where multiple outcomes are possible.
        # K for $\pi$ mixing coefficients
        # K * latetent_dimension for the means
        # K * latetent_dimension for the standard deviations

        self.fc_pi = nn.Linear(hidden_units, num_mixtures)
        self.fc_mu = nn.Linear(hidden_units, num_mixtures * latent_dimension)
        self.fc_log_sigma = nn.Linear(hidden_units, num_mixtures * latent_dimension)

    def forward(self, latents, actions, hidden, temperature=1.0) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        actions = self.action_embedding(actions.long())
        # Latents conditioned by the actions
        rnn_out, hidden = self.rnn(
            torch.cat(
                [latents, actions],
                dim=-1,
            ),
            hidden,
        )
        # MDN layer: outputs mixture weights (pi), means (mu), variances (sigma)
        pi = self.fc_pi(rnn_out)
        mu = self.fc_mu(rnn_out)
        log_sigma = self.fc_log_sigma(rnn_out)

        # As described in: https://arxiv.org/pdf/1308.0850 (19,20)
        # The mixture weight outputs are normalised with a softmax
        # function to ensure they form a valid discrete distribution,
        # and the other outputs are passed through suitable functions to keep
        # their values within meaningful range (for example the exponential function is
        # typically applied to outputs used as scale parameters, which must be positive)
        # As described in: https://arxiv.org/pdf/1704.03477 (4)
        # We can control the level of randomness we would like our samples to have during the
        # sampling process by introducing a temperature parameter $\tau$.
        # We can scale the softmax parameters of the categorial distribution
        # and also the $\sigma$ parameters of the bivariate normal distribution
        # by a temperature parameter $\tau$ , to control the level of randomness in our samples.
        pi = F.softmax(pi / temperature, dim=-1)
        # The sigmas are exponentiated, because this guarantees positivity
        sigma = torch.exp(log_sigma)
        return pi, mu, sigma, hidden

    def __loss(self, pi, mu, sigma, target):
        # Expand target to match the dimensions of mu and sigma
        target = target.unsqueeze(1)  # Shape: (batch_size, 1, latent_dim)
        # Reshape mu and sigma to separate mixture components and latent dimensions
        batch_size = target.size(0)
        num_mixtures = self.num_mixtures
        latent_dim = target.size(-1)

        # Reshape mu and sigma to [batch_size, num_mixtures, latent_dimension]
        mu = mu.view(batch_size, num_mixtures, latent_dim)
        sigma = sigma.view(batch_size, num_mixtures, latent_dim)
        # Create normal distributions for each mixture component
        normal = torch.distributions.Normal(loc=mu, scale=sigma)
        # Compute log probabilities for each component
        log_probs = normal.log_prob(
            target
        )  # Shape: (batch_size, num_mixtures, latent_dim)

        # Sum over the latent dimensions to get total log probability per component
        log_probs = log_probs.sum(dim=-1)  # Shape: (batch_size, num_mixtures)

        # Weight by the mixture probabilities (log-sum-exp trick for numerical stability)
        log_probs = torch.logsumexp(
            torch.log(pi) + log_probs, dim=-1
        )  # Shape: (batch_size)

        # Negative log likelihood
        return -log_probs.mean()

    def train(
        self,
        dataloader: RolloutDataloader,
        conv_vae: ConvVAE,
        epochs: int = 10,
    ):
        super().train()
        conv_vae.eval()
        optimizer = torch.optim.Adam(self.parameters())

        for epoch in range(epochs):
            epoch_loss = 0
            hidden = None
            for batch_rollouts_observations, batch_rollouts_actions, _ in dataloader:
                batch_rollouts_observations = batch_rollouts_observations.to(
                    next(self.parameters()).device
                )
                batch_rollouts_observations = batch_rollouts_observations.reshape(
                    batch_rollouts_observations.shape[1],
                    batch_rollouts_observations.shape[0],
                    *batch_rollouts_observations.shape[2:],
                )
                batch_rollouts_actions = batch_rollouts_actions.to(
                    next(self.parameters()).device
                )
                batch_rollouts_actions = batch_rollouts_actions.reshape(
                    batch_rollouts_actions.shape[1],
                    batch_rollouts_actions.shape[0],
                    *batch_rollouts_actions.shape[2:],
                )

                optimizer.zero_grad()
                batch_loss = torch.zeros([])
                hidden = None
                for timestep_observation, timestep_action in zip(
                    batch_rollouts_observations, batch_rollouts_actions
                ):

                    optimizer.zero_grad()
                    target = conv_vae.get_latent(timestep_observation)
                    pi, mu, sigma, hidden = self(target, timestep_action, hidden)
                    loss = self.__loss(pi, mu, sigma, target)
                    batch_loss += loss
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()

        torch.save(self.state_dict(), Path("models") / "memory.pt")


if __name__ == "__main__":
    file_path = Path("data") / "dataset.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if file_path.exists():
        dataset = RolloutDataset.load(file_path=file_path)
    else:
        dataset = RolloutDataset(num_rollouts=10, max_steps=10)
        dataset.save(file_path=file_path)
    dataloader = RolloutDataloader(dataset, 2)
    vision = ConvVAE()  # .from_pretrained(file_path=Path("models/vision.pt"))
    memory = MDN_RNN()
    memory.train(dataloader, vision)
    # for (
    #     batch_rollouts_observations,
    #     batch_rollouts_actions,
    #     _,
    # ) in dataloader:
    #     latents = conv_vae.get_latents(batch_rollouts_observations)
    #     print(f"{latents.shape=}")
    #     memory = MDN_RNN()
    #     print(f"{batch_rollouts_actions.shape}")
    #     memory(latents, batch_rollouts_actions, None)
