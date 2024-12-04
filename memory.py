import os
from pathlib import Path
from typing import Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim.adam
from dataset import RolloutDataloader, RolloutDataset
from vision import ConvVAE
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image, ImageDraw
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Create GIF for visual inspection
def create_gif(
    episode,
    vision,
    memory,
    save_path=Path("memory_reconstruction.gif"),
):
    observations = episode.observations.unsqueeze(0).to(DEVICE)
    actions = episode.actions.unsqueeze(0).to(DEVICE)
    latents = vision.get_latents(observations=observations)
    pi, mu, sigma, _ = memory(latents[:, :-1, :], actions[:, :-1])
    predicted_latents = memory.sample_latent(
        pi.squeeze(0), mu.squeeze(0), sigma.squeeze(0)
    )  # Shape: (399, 32)

    # Decode latent vectors to reconstruct images
    vae_reconstructions = vision.decoder(latents.squeeze(0))  # Shape: (400, 3, 64, 64)
    mdn_reconstructions = vision.decoder(predicted_latents)
    scale_factor = 1  # Scale images for better resolution
    spacing = 1  # Padding between images
    img_width, img_height = 64 * scale_factor, 64 * scale_factor
    total_width = img_width * 3 + spacing * 2  # 3 images side-by-side
    total_height = img_height

    images = []
    for t in range(mdn_reconstructions.shape[0]):  # Up to the length of MDN outputs
        # Original observation
        original_img = T.Resize((img_height, img_width))(
            T.ToPILImage()(observations[0, t].cpu())
        )

        # VAE reconstruction
        vae_img = T.Resize((img_height, img_width))(
            T.ToPILImage()(vae_reconstructions[t].cpu())
        )

        # MDN reconstruction
        mdn_img = T.Resize((img_height, img_width))(
            T.ToPILImage()(mdn_reconstructions[t].cpu())
        )

        # Combine images with padding
        combined_img = Image.new("RGB", (total_width, total_height), (0, 0, 0))
        combined_img.paste(original_img, (0, 0))
        combined_img.paste(vae_img, (img_width + spacing, 0))
        combined_img.paste(mdn_img, (2 * (img_width + spacing), 0))

        images.append(combined_img)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Save as GIF
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=200,  # Increase duration for slower playback
        loop=0,
    )
    print(f"Reconstruction GIF saved to {save_path}")


class MDN_RNN(nn.Module):
    """
    Mixture Density Network with LSTM for modeling latent dynamics.

    This model combines an LSTM with Mixture Density Network (MDN) output layers to predict
    a mixture of Gaussian distributions over latent states conditioned on inputs.

    Args:
        latent_dimension (int): Dimensionality of the latent space. Default is 32.
        hidden_units (int): Number of hidden units in the LSTM. Default is 256.
        num_mixtures (int): Number of mixture components in the MDN. Default is 5.
    """

    def __init__(
        self, latent_dimension: int = 32, hidden_units: int = 256, num_mixtures: int = 5
    ):
        super().__init__()
        self.hidden_dim = hidden_units
        self.num_mixtures = num_mixtures
        self.latent_dimension = latent_dimension
        self.rnn = nn.LSTM(latent_dimension + 1, hidden_units, batch_first=True)
        self.fc_pi = nn.Linear(hidden_units, num_mixtures)
        self.fc_mu = nn.Linear(hidden_units, num_mixtures * latent_dimension)
        self.fc_log_sigma = nn.Linear(hidden_units, num_mixtures * latent_dimension)

    def forward(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]
    ]:
        """
        Forward pass through the MDN_RNN.

        Args:
            latents (torch.Tensor): Input latent states of shape
                (batch_size, seq_len, latent_dimension).
            actions (torch.Tensor): Actions of shape (batch_size, seq_len).
            hidden (Optional[Tuple[torch.Tensor, torch.Tensor]]):
                Initial LSTM hidden state and cell state.

        Returns:
            Tuple:
                - pi (torch.Tensor): Mixture weights of shape (batch_size, seq_len, num_mixtures).
                - mu (torch.Tensor): Means of the Gaussians of shape
                    (batch_size, seq_len, num_mixtures * latent_dimension).
                - sigma (torch.Tensor): Standard deviations of the Gaussians of shape
                    (batch_size, seq_len, num_mixtures * latent_dimension).
                - hidden (Tuple[torch.Tensor, torch.Tensor]): Updated LSTM hidden and cell states.
        """
        actions = actions.unsqueeze(-1)  # Expand action dimension
        rnn_out, hidden = self.rnn(
            torch.cat([latents, actions], dim=-1),
            hidden,
        )

        # MDN output layers
        pi = F.softmax(self.fc_pi(rnn_out), dim=-1)
        mu = self.fc_mu(rnn_out)
        log_sigma = self.fc_log_sigma(rnn_out)
        sigma = torch.exp(log_sigma)
        return pi, mu, sigma, hidden  # type: ignore

    def loss(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the MDN loss function.

        Args:
            pi (torch.Tensor): Mixture weights of shape (batch_size, seq_len, num_mixtures).
            mu (torch.Tensor): Means of the Gaussians of shape (batch_size, seq_len, num_mixtures * latent_dimension).
            sigma (torch.Tensor): Standard deviations of the Gaussians of shape (batch_size, seq_len, num_mixtures * latent_dimension).
            target (torch.Tensor): Target latent states of shape (batch_size, seq_len, latent_dimension).

        Returns:
            torch.Tensor: Mean MDN loss over the batch.
        """
        batch_size, seq_len = target.shape[:2]
        mu = mu.view(batch_size, seq_len, self.num_mixtures, self.latent_dimension)
        sigma = sigma.view(
            batch_size, seq_len, self.num_mixtures, self.latent_dimension
        )
        target = target.unsqueeze(2).expand(-1, -1, self.num_mixtures, -1)
        sigma = torch.clamp(sigma, min=1e-4)

        normal = torch.distributions.Normal(loc=mu, scale=sigma)
        log_probs = normal.log_prob(target).sum(dim=-1)
        log_pi = torch.log(pi + 1e-4)
        log_probs = -torch.logsumexp(log_pi + log_probs, dim=-1)
        return log_probs.mean()

    def sample_latent(
        self, pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Samples latent states from the predicted mixture of Gaussians.

        Args:
            pi (torch.Tensor): Mixture weights of shape (batch_size, num_mixtures).
            mu (torch.Tensor): Means of the Gaussians of shape (batch_size, latent_dimension * num_mixtures).
            sigma (torch.Tensor): Standard deviations of the Gaussians of shape (batch_size, latent_dimension * num_mixtures).

        Returns:
            torch.Tensor: Sampled latent states of shape (batch_size, latent_dimension).
        """
        batch_size = mu.size(0)
        mu = mu.view(batch_size, self.latent_dimension, self.num_mixtures)
        sigma = sigma.view(batch_size, self.latent_dimension, self.num_mixtures)

        categorical = torch.distributions.Categorical(pi)
        mixture_indices = categorical.sample()  # Shape: (batch_size,)
        mixture_indices = mixture_indices.unsqueeze(1).expand(-1, self.latent_dimension)

        selected_mu = torch.gather(mu, 2, mixture_indices.unsqueeze(-1)).squeeze(-1)
        selected_sigma = torch.gather(sigma, 2, mixture_indices.unsqueeze(-1)).squeeze(
            -1
        )

        normal = torch.distributions.Normal(selected_mu, selected_sigma)
        return normal.rsample()

    @staticmethod
    def from_pretrained(file_path: Path = Path("models") / "memory.pt") -> "MDN_RNN":
        """
        Loads a pretrained MDN_RNN model from a file.

        Args:
            file_path (Path): Path to the saved model file.

        Returns:
            MDN_RNN: Loaded MDN_RNN instance.
        """
        loaded_data = torch.load(file_path, weights_only=True)
        mdn_rnn = MDN_RNN()
        mdn_rnn.load_state_dict(loaded_data)
        return mdn_rnn


class MemoryTrainer:
    """
    Trainer class for training and evaluating an MDN_RNN model
    using precomputed latents from a ConvVAE.
    """

    def _train_step(
        self,
        memory: MDN_RNN,
        vision: ConvVAE,
        train_dataloader: RolloutDataloader,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Perform a single training step.

        Args:
            memory (MDN_RNN): The MDN_RNN model to train.
            vision (ConvVAE): The ConvVAE model used for latent space encoding.
            train_dataloader (RolloutDataloader): Dataloader for the training dataset.
            optimizer (torch.optim.Optimizer): Optimizer for training.

        Returns:
            float: Average training loss for the step.
        """
        memory.train()
        train_loss = 0

        for batch_rollouts_observations, batch_rollouts_actions, _ in train_dataloader:

            # Move data to the correct device
            device = next(memory.parameters()).device
            batch_rollouts_observations = batch_rollouts_observations.to(device)
            batch_rollouts_actions = batch_rollouts_actions.to(device)

            # Precompute latent vectors
            latent_vectors = vision.get_latents(batch_rollouts_observations)

            # Target is the next latent vector
            target = latent_vectors[:, 1:, :]

            # Predict using MDN-RNN
            pi, mu, sigma, _ = memory(
                latent_vectors[:, :-1],  # Input latent vectors
                batch_rollouts_actions[:, :-1],  # Input actions
            )

            # Compute loss and backpropagate
            loss = memory.loss(pi, mu, sigma, target)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        torch.cuda.empty_cache()
        return train_loss

    def _test_step(
        self,
        memory: MDN_RNN,
        vision: ConvVAE,
        test_dataloader: RolloutDataloader,
    ) -> float:
        """
        Perform a single evaluation step.

        Args:
            memory (MDN_RNN): The MDN_RNN model to evaluate.
            vision (ConvVAE): The ConvVAE model used for latent space encoding.
            test_dataloader (RolloutDataloader): Dataloader for the test dataset.

        Returns:
            float: Average test loss for the step.
        """
        memory.eval()
        test_loss = 0

        for batch_rollouts_observations, batch_rollouts_actions, _ in test_dataloader:
            # Move data to the correct device
            device = next(memory.parameters()).device
            batch_rollouts_observations = batch_rollouts_observations.to(device)
            batch_rollouts_actions = batch_rollouts_actions.to(device)

            # Precompute latent vectors
            latent_vectors = vision.get_latents(batch_rollouts_observations)

            # Target is the next latent vector
            target = latent_vectors[:, 1:, :]

            # Predict using MDN-RNN
            pi, mu, sigma, _ = memory(
                latent_vectors[:, :-1],  # Input latent vectors
                batch_rollouts_actions[:, :-1],  # Input actions
            )

            # Compute loss
            loss = memory.loss(pi, mu, sigma, target)
            test_loss += loss.item()

        test_loss /= len(test_dataloader)
        torch.cuda.empty_cache()
        return test_loss

    def train(
        self,
        memory: MDN_RNN,
        vision: ConvVAE,
        train_dataloader: RolloutDataloader,
        test_dataloader: RolloutDataloader,
        optimizer: torch.optim.Optimizer,
        epochs: int = 10,
        save_path: Path = Path("models") / "memory.pt",
    ):
        """
        Train the MDN_RNN model for a specified number of epochs.

        Args:
            memory (MDN_RNN): The MDN_RNN model to train.
            vision (ConvVAE): The ConvVAE model for latent space encoding.
            train_dataloader (RolloutDataloader): Dataloader for the training dataset.
            test_dataloader (RolloutDataloader): Dataloader for the test dataset.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            epochs (int): Number of epochs to train for.
            save_path (Path): Path to save the trained model.
        """
        vision.eval()  # Ensure vision is in evaluation mode
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            train_loss = self._train_step(memory, vision, train_dataloader, optimizer)
            test_loss = 0  # self._test_step(memory, vision, test_dataloader)
            # scheduler.step()

            print(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}"
            )

        # Save the model
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(memory.state_dict(), save_path)
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    # Path to save or load the dataset
    file_path = Path("data") / "dataset.pt"

    # Load or create the dataset
    if file_path.exists():
        dataset = RolloutDataset.load(file_path=file_path)
    else:
        dataset = RolloutDataset(num_rollouts=20, max_steps=10)
        dataset.save(file_path=file_path)

    # Split dataset into training, testing, and evaluation sets
    train_rollouts, test_rollouts, eval_rollouts = torch.utils.data.random_split(
        dataset, [0.5, 0.3, 0.2]
    )

    # Create new RolloutDataset instances for each split
    training_set = RolloutDataset(episodes=train_rollouts.dataset.episodes)  # type: ignore
    test_set = RolloutDataset(episodes=test_rollouts.dataset.episodes)  # type: ignore
    eval_set = RolloutDataset(episodes=eval_rollouts.dataset.episodes)  # type: ignore

    # Initialize dataloaders
    train_dataloader = RolloutDataloader(training_set, batch_size=32)
    test_dataloader = RolloutDataloader(test_set, batch_size=32)

    # Load pretrained models
    vision = ConvVAE().from_pretrained().to(DEVICE)
    # memory = MDN_RNN().from_pretrained().to(DEVICE)
    memory = MDN_RNN().to(DEVICE)
    memory_trainer = MemoryTrainer()
    memory_trainer.train(
        memory,
        vision,
        train_dataloader,
        test_dataloader,
        torch.optim.Adam(memory.parameters()),
        10,
    )

    for i in range(10):
        create_gif(
            dataset.episodes[i],
            vision,
            memory,
            save_path=Path("memory_reconstructions") / f"reconstruction{i}.gif",
        )
