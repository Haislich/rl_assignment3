import os
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from dataset import RolloutDataloader, RolloutDataset
from vision import ConvVAE
import matplotlib.pyplot as plt
from PIL import Image

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
        # The *2 is not taken from the original paper, the idea is the following.
        # We want to condition the latent on the actions, however while latents have 32 dimensions
        # actions are scalars, so actions are embedded in a latest_dimensional space,
        # This will result in actions and observation have an equal weight in the resulting LSTM.
        self.rnn = nn.LSTM(latent_dimension + 1, hidden_units, batch_first=True)
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

        # actions = self.action_embedding(actions.long())
        actions = actions.unsqueeze(-1)
        rnn_out, hidden = self.rnn(
            torch.cat(
                [latents, actions],
                dim=-1,
            ),
            hidden,
        )

        # MDN layer: outputs mixture weights (pi), means (mu), variances (sigma)
        pi = self.fc_pi(rnn_out)  # Use the final layer's hidden state
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

        batch_size = target.shape[0]
        seq_len = target.shape[1]
        mu = mu.view(batch_size, seq_len, self.num_mixtures, self.latent_dimension)
        sigma = sigma.view(
            batch_size, seq_len, self.num_mixtures, self.latent_dimension
        )
        target = target.unsqueeze(2).expand(-1, -1, mu.shape[2], -1)

        # Stabilize sigma
        sigma = torch.clamp(sigma, min=1e-8)

        # Compute log probabilities
        normal = torch.distributions.Normal(loc=mu, scale=sigma)
        log_probs = normal.log_prob(target)
        log_probs = log_probs.sum(dim=-1)  # Shape: (batch_size, num_mixtures)

        # Stabilize pi
        log_pi = torch.log(pi + 1e-8)

        # Compute log-sum-exp
        log_probs = log_pi + log_probs
        log_probs = -torch.logsumexp(log_probs, dim=-1)  # Shape: (batch_size)

        return log_probs.mean()

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
        hidden = None
        for batch_rollouts_observations, batch_rollouts_actions, _ in train_dataloader:

            # Move data to the correct device and reshape
            batch_rollouts_observations = batch_rollouts_observations.to(
                next(memory.parameters()).device
            )
            batch_rollouts_actions = batch_rollouts_actions.to(
                next(memory.parameters()).device
            )

            # Precompute latent vectors for the entire sequence
            latent_vectors = vision.get_latents(batch_rollouts_observations)

            batch_loss = torch.zeros([]).to(next(memory.parameters()).device)

            target = latent_vectors[:, 1:, :]  # Next latent vector

            # Predict using MDN-RNN
            pi, mu, sigma, _ = memory(
                latent_vectors[:, :-1], batch_rollouts_actions[:, :-1], hidden
            )

            # Compute loss
            loss = memory.loss(pi, mu, sigma, target)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(memory.parameters(), max_norm=10.0)

            # # Log gradient statistics
            # print("Gradient statistics:")
            # for name, param in memory.named_parameters():
            #     if param.grad is not None:
            #         print(
            #             f"{name} gradient mean: {param.grad.mean().item():.5f}, std: {param.grad.std().item():.5f}"
            #         )

            # Update weights
            optimizer.step()
            train_loss += loss.item()

        # Average train loss
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
            # test_loss = self._test_step(
            #     memory,
            #     vision,
            #     test_dataloader,
            # )
            print(f"Epoch {epoch} | {train_loss=} ")
        os.makedirs(save_path.parents[0], exist_ok=True)
        torch.save(memory.state_dict(), save_path)


if __name__ == "__main__":
    file_path = Path("data") / "dataset.pt"

    if file_path.exists():
        dataset = RolloutDataset.load(file_path=file_path)
    else:
        dataset = RolloutDataset(num_rollouts=20, max_steps=10)
        dataset.save(file_path=file_path)

    train_rollouts, test_rollouts, eval_rollouts = torch.utils.data.random_split(
        dataset, [0.5, 0.3, 0.2]
    )
    training_set = RolloutDataset(rollouts=train_rollouts.dataset.rollouts)  # type: ignore
    test_set = RolloutDataset(rollouts=test_rollouts.dataset.rollouts)  # type: ignore
    eval_set = RolloutDataset(rollouts=eval_rollouts.dataset.rollouts)  # type: ignore

    train_dataloader = RolloutDataloader(training_set, batch_size=2)
    test_dataloader = RolloutDataloader(test_set, batch_size=2)

    vision = ConvVAE().from_pretrained().to(DEVICE)
    memory = MDN_RNN().to(DEVICE)
    # memory_trainer = MemoryTrainer()
    # memory_trainer.train(
    #     memory,
    #     vision,
    #     train_dataloader,
    #     test_dataloader,
    #     torch.optim.Adam(memory.parameters()),
    #     epochs=50,
    # )
    # exit()
    memory = MDN_RNN().from_pretrained().to(DEVICE)
    vision.eval()
    memory.eval()
    # Unpack the dataloader's batch
    observations, actions, rewards = next(iter(test_dataloader))

    # Move tensors to the correct device
    observations = observations.to(DEVICE)  # Shape: (batch_size, timesteps, 3, 64, 64)
    actions = actions.to(DEVICE)  # Shape: (batch_size, timesteps)
    rewards = rewards.to(DEVICE)  # Shape: (batch_size, timesteps)

    # Iterate over timesteps (2nd dimension of the tensors)
    batch_size, timesteps, _, _, _ = observations.shape
    hidden = None
    mdn_reconstructions = []
    vae_reconstructions = []

    for t in range(timesteps):
        # Extract the t-th timestep across the batch
        timestep_observations = observations[
            :, t, :, :, :
        ]  # Shape: (batch_size, 3, 64, 64)
        timestep_actions = actions[:, t]  # Shape: (batch_size,)

        # Pass observations through the VAE to get target latents
        target_latent = vision.get_latent(
            timestep_observations
        )  # Shape: (batch_size, latent_dim)

        # Pass target latents and actions through the MDN_RNN
        pi, mu, sigma, hidden = memory(
            target_latent, timestep_actions, hidden
        )  # MDN outputs

        # Sample from the MDN to get the predicted latent vector
        sampled_latent = memory.sample_latent(pi, mu, sigma)

        # Decode the target and sampled latents using the VAE
        vae_reconstruction = (
            vision.decoder(target_latent).detach().cpu()
        )  # From VAE target latent
        mdn_reconstruction = (
            vision.decoder(sampled_latent).detach().cpu()
        )  # From MDN sampled latent

        vae_reconstructions.append(
            vae_reconstruction[0]
        )  # Keep the first sample in the batch
        mdn_reconstructions.append(
            mdn_reconstruction[0]
        )  # Keep the first sample in the batch

    vae_reconstructions = torch.stack(
        vae_reconstructions
    ).numpy()  # Shape: (timesteps, C, H, W)
    mdn_reconstructions = torch.stack(
        mdn_reconstructions
    ).numpy()  # Shape: (timesteps, C, H, W)

    # Create a list to store images for the GIF
    frames = []

    for t in range(timesteps):
        # Original Observation
        original = (observations[0, t].permute(1, 2, 0).cpu().numpy() * 255).astype(
            "uint8"
        )  # Normalize to [0, 255]

        # VAE Reconstruction
        vae_image = (vae_reconstructions[t].transpose(1, 2, 0) * 255).astype(
            "uint8"
        )  # Convert to HWC and normalize

        # MDN Reconstruction
        mdn_image = (mdn_reconstructions[t].transpose(1, 2, 0) * 255).astype(
            "uint8"
        )  # Convert to HWC and normalize

        # Create a single canvas for all three images
        h, w, c = original.shape
        canvas = Image.new("RGB", (w * 3, h))  # Width: 3 images side by side
        canvas.paste(Image.fromarray(original), (0, 0))
        canvas.paste(Image.fromarray(vae_image), (w, 0))
        canvas.paste(Image.fromarray(mdn_image), (w * 2, 0))

        # Add to frames
        frames.append(canvas)

    # Save frames as a GIF
    frames[0].save(
        "reconstructions.gif",
        save_all=True,
        append_images=frames[1:],
        duration=200,  # Duration between frames in milliseconds
        loop=0,  # Infinite loop
    )

    print("GIF saved as reconstructions.gif")
