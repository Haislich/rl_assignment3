import os
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from dataset import RolloutDataloader, RolloutDataset
from vision import ConvVAE
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Credits:
# - https://github.com/sksq96/pytorch-mdn/blob/master/mdn-rnn.ipynb
# - https://github.com/hardmaru/WorldModelsExperiments/blob/master/carracing/rnn/rnn.py
#
class MDN_RNN(nn.Module):
    def __init__(self, latent_dimension=32, hidden_units=256, num_mixtures=5):
        super().__init__()
        self.hidden_dim = hidden_units
        self.num_mixtures = num_mixtures
        self.latent_dimension = latent_dimension
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

    def loss(self, pi, mu, sigma, target):
        # Expand target to match the dimensions of mu and sigma
        target = target.unsqueeze(1)  # Shape: (batch_size, 1, latent_dim)
        # Reshape mu and sigma to separate mixture components and latent dimensions
        batch_size = target.size(0)

        # Reshape mu and sigma to [batch_size, num_mixtures, latent_dimension]
        mu = mu.view(batch_size, self.num_mixtures, self.latent_dimension)
        sigma = sigma.view(batch_size, self.num_mixtures, self.latent_dimension)
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

    def sample_latent(self, pi, mu, sigma):
        batch_size = mu.size(0)

        mu = mu.view(batch_size, self.latent_dimension, self.num_mixtures)
        sigma = sigma.view(batch_size, self.latent_dimension, self.num_mixtures)

        # Sample a mixture component index for each item in the batch
        categorical = torch.distributions.Categorical(pi)
        mixture_indices = categorical.sample()  # Shape: (batch_size,)
        # Expand mixture_indices for latent_dimension
        mixture_indices = mixture_indices.unsqueeze(1).expand(-1, self.latent_dimension)

        # Gather the corresponding mu and sigma for the selected mixture component
        selected_mu = torch.gather(mu, 2, mixture_indices.unsqueeze(-1)).squeeze(-1)
        selected_sigma = torch.gather(sigma, 2, mixture_indices.unsqueeze(-1)).squeeze(
            -1
        )

        # Sample from the Gaussian defined by selected_mu and selected_sigma
        normal = torch.distributions.Normal(selected_mu, selected_sigma)
        sampled_latent = normal.rsample()  # Shape: (batch_size, latent_dim)
        return sampled_latent

    @staticmethod
    def from_pretrained(file_path: Path = Path("models") / "memory.pt"):
        loaded_data = torch.load(file_path, weights_only=False)
        mdn_rnn = MDN_RNN()
        mdn_rnn.load_state_dict(loaded_data)
        return mdn_rnn


class MemoryTrainer:
    def _train_step(
        self,
        memory: MDN_RNN,
        vision: ConvVAE,
        train_dataloader: RolloutDataloader,
        optimizer: torch.optim.Optimizer,
    ):
        memory.train()
        train_loss = 0
        for batch_rollouts_observations, batch_rollouts_actions, _ in train_dataloader:
            batch_rollouts_observations = batch_rollouts_observations.to(
                next(memory.parameters()).device
            )
            batch_rollouts_observations = batch_rollouts_observations.permute(
                1, 0, 2, 3, 4
            )
            batch_rollouts_actions = batch_rollouts_actions.to(
                next(memory.parameters()).device
            )
            batch_rollouts_actions = batch_rollouts_actions.permute(1, 0)

            batch_loss = torch.zeros([]).to(next(memory.parameters()).device)
            hidden = None

            for timestep_observation, timestep_action in zip(
                batch_rollouts_observations, batch_rollouts_actions
            ):
                target = vision.get_latent(timestep_observation)
                pi, mu, sigma, hidden = memory(target, timestep_action, hidden)
                rnn_latent = memory.sample_latent(pi, mu, sigma)
                loss = memory.loss(pi, mu, sigma, target)
                batch_loss += loss

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()
        train_loss /= len(train_dataloader)
        return train_loss

    def _test_step(
        self,
        memory: MDN_RNN,
        vision: ConvVAE,
        test_dataloader: RolloutDataloader,
    ):
        memory.eval()
        test_loss = 0
        for batch_rollouts_observations, batch_rollouts_actions, _ in test_dataloader:
            batch_rollouts_observations = batch_rollouts_observations.to(
                next(memory.parameters()).device
            )
            batch_rollouts_observations = batch_rollouts_observations.permute(
                1, 0, 2, 3, 4
            )
            batch_rollouts_actions = batch_rollouts_actions.to(
                next(memory.parameters()).device
            )
            batch_rollouts_actions = batch_rollouts_actions.permute(1, 0)
            batch_loss = torch.zeros([]).to(next(memory.parameters()).device)
            hidden = None
            for timestep_observation, timestep_action in zip(
                batch_rollouts_observations, batch_rollouts_actions
            ):

                target = vision.get_latent(timestep_observation)
                pi, mu, sigma, hidden = memory(target, timestep_action, hidden)
                loss = memory.loss(pi, mu, sigma, target)
                batch_loss += loss
            test_loss += batch_loss.item()
        test_loss /= len(test_dataloader)
        return test_loss

    def train(
        self,
        memory: MDN_RNN,
        vision: ConvVAE,
        train_dataloader: RolloutDataloader,
        test_dataloader: RolloutDataloader,
        optimizer: torch.optim.Optimizer,
        epochs: int = 10,
        save_path=Path("models") / "memory.pt",
    ):
        vision.eval()

        for epoch in range(epochs):
            train_loss = self._train_step(
                memory,
                vision,
                train_dataloader,
                optimizer,
            )
            test_loss = self._test_step(
                memory,
                vision,
                test_dataloader,
            )
            print(f"Epoch {epoch} | {train_loss=} | {test_loss=}")
        os.makedirs(save_path.parents[0], exist_ok=True)
        torch.save(memory.state_dict(), save_path)


if __name__ == "__main__":
    file_path = Path("data") / "dataset.pt"

    if file_path.exists():
        dataset = RolloutDataset.load(file_path=file_path)
    else:
        dataset = RolloutDataset(num_rollouts=1000, max_steps=200)
        dataset.save(file_path=file_path)

    train_rollouts, test_rollouts, eval_rollouts = torch.utils.data.random_split(
        dataset, [0.5, 0.3, 0.2]
    )
    training_set = RolloutDataset(rollouts=train_rollouts.dataset.rollouts)  # type: ignore
    test_set = RolloutDataset(rollouts=test_rollouts.dataset.rollouts)  # type: ignore
    eval_set = RolloutDataset(rollouts=eval_rollouts.dataset.rollouts)  # type: ignore

    train_dataloader = RolloutDataloader(training_set, batch_size=1)
    test_dataloader = RolloutDataloader(test_set, batch_size=1)

    vision = ConvVAE().from_pretrained().to(DEVICE)

    memory = MDN_RNN().from_pretrained().to(DEVICE)
    # memory_trainer = MemoryTrainer()

    # memory_trainer.train(
    #     memory,
    #     vision,
    #     train_dataloader,
    #     test_dataloader,
    #     torch.optim.Adam(memory.parameters()),
    # )
    vision.eval()
    memory.eval()
    observations, actions, _ = next(iter(test_dataloader))  # Get one batch
    observations = observations.to(DEVICE)  # Shape: (timesteps, batch_size, C, H, W)
    actions = actions.to(DEVICE)  # Shape: (timesteps, batch_size)

    # Process observations and actions
    hidden = None
    mdn_reconstructions = []
    vae_reconstructions = []

    for timestep_observation, timestep_action in zip(observations, actions):
        # Pass through the VAE to get the target latent
        target_latent = vision.get_latent(
            timestep_observation
        )  # Shape: (batch_size, latent_dim)

        # Pass through MDN_RNN
        pi, mu, sigma, hidden = memory(
            target_latent, timestep_action, hidden
        )  # Get MDN params

        # Sample from the MDN
        sampled_latent = memory.sample_latent(pi, mu, sigma)

        # Decode from the VAE
        vae_reconstruction = (
            vision.decoder(target_latent).detach().cpu()
        )  # From target latent
        mdn_reconstruction = (
            vision.decoder(sampled_latent).detach().cpu()
        )  # From MDN latent

        vae_reconstructions.append(vae_reconstruction[0])  # Keep first batch image
        mdn_reconstructions.append(mdn_reconstruction[0])  # Keep first batch image

    # Convert to numpy for plotting
    vae_reconstructions = torch.stack(
        vae_reconstructions
    ).numpy()  # Shape: (timesteps, C, H, W)
    mdn_reconstructions = torch.stack(
        mdn_reconstructions
    ).numpy()  # Shape: (timesteps, C, H, W)

    # Plot the results
    timesteps = len(vae_reconstructions)
    for t in range(timesteps):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Original Observation
        original = (
            observations[t].permute(1, 2, 0).cpu().numpy()
        )  # Permute for HWC format
        axes[0].imshow(original)
        axes[0].set_title("Original Observation")
        axes[0].axis("off")

        # VAE Reconstruction
        vae_image = vae_reconstructions[t].transpose(1, 2, 0)  # Convert to HWC
        axes[1].imshow(vae_image)
        axes[1].set_title("VAE Reconstruction")
        axes[1].axis("off")

        # MDN Reconstruction
        mdn_image = mdn_reconstructions[t].transpose(1, 2, 0)  # Convert to HWC
        axes[2].imshow(mdn_image)
        axes[2].set_title("MDN Reconstruction")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()
