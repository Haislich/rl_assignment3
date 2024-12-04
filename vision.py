"""Vision component"""

from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from dataset import RolloutDataloader, RolloutDataset
import torchvision.transforms as T
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_gif(
    episode,
    vision,
    save_path=Path("vae_reconstruction.gif"),
):
    observations = episode.observations.unsqueeze(0).to(DEVICE)
    latents = vision.get_latents(observations=observations)

    # Decode latent vectors to reconstruct images
    vae_reconstructions = vision.decoder(latents.squeeze(0))
    scale_factor = 1  # Scale images for better resolution
    spacing = 1  # Padding between images
    img_width, img_height = 64 * scale_factor, 64 * scale_factor
    total_width = img_width * 2 + spacing * 2  # 3 images side-by-side
    total_height = img_height

    images = []
    for t in range(vae_reconstructions.shape[0]):  # Up to the length of MDN outputs
        # Original observation
        original_img = T.Resize((img_height, img_width))(
            T.ToPILImage()(observations[0, t].cpu())
        )
        # VAE reconstruction
        vae_img = T.Resize((img_height, img_width))(
            T.ToPILImage()(vae_reconstructions[t].cpu())
        )

        # Combine images with padding
        combined_img = Image.new("RGB", (total_width, total_height), (0, 0, 0))
        combined_img.paste(original_img, (0, 0))
        combined_img.paste(vae_img, (img_width + spacing, 0))
        images.append(combined_img)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Save as GIF
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=20,  # Increase duration for slower playback
        loop=0,
    )
    print(f"Vae reconstruction GIF saved to {save_path}")


class Encoder(nn.Module):
    """
    Convolutional Encoder for generating latent representations.

    This encoder uses a series of convolutional layers to process input images and
    produces mean (`mu`) and log-variance (`log_sigma`) vectors representing the
    latent distribution.

    Args:
        latent_dimension (int): Dimension of the latent space.
        stride (int): Stride for convolutional layers. Default is 2.
    """

    def __init__(self, latent_dimension: int, *, stride: int = 2):
        super().__init__()
        self.relu_conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=stride)
        self.relu_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=stride)
        self.relu_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=stride)
        self.relu_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=stride)

        # Fully connected layers for latent distribution
        self.fc_mu = nn.Linear(2 * 2 * 256, latent_dimension)
        self.fc_sigma = nn.Linear(2 * 2 * 256, latent_dimension)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Mean (`mu`) and log-variance (`log_sigma`)
            tensors of shape (batch_size, latent_dimension).
        """
        x = F.relu(self.relu_conv1(x))
        x = F.relu(self.relu_conv2(x))
        x = F.relu(self.relu_conv3(x))
        x = F.relu(self.relu_conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        log_sigma = self.fc_sigma(x)

        return mu, log_sigma


class Decoder(nn.Module):
    """
    Convolutional Decoder for reconstructing images from latent representations.

    This decoder takes a latent vector and applies transposed convolution layers
    to generate an output image.

    Args:
        latent_dimension (int): Dimension of the latent space.
        image_channels (int): Number of channels in the output image (e.g., 3 for RGB).
        stride (int): Stride for transposed convolution layers. Default is 2.
    """

    def __init__(self, latent_dimension: int, image_channels: int, *, stride: int = 2):
        super().__init__()
        self.fc = nn.Linear(latent_dimension, 1024)
        self.relu_deconv1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=stride)
        self.relu_deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=stride)
        self.relu_deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=stride)
        self.sigmoid_deconv = nn.ConvTranspose2d(
            32, image_channels, kernel_size=6, stride=stride
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Args:
            x (torch.Tensor): Input latent tensor of shape (batch_size, latent_dimension).

        Returns:
            torch.Tensor: Reconstructed image tensor of shape
                          (batch_size, image_channels, height, width).
        """
        x = self.fc(x)  # Fully connected layer
        x = x.unsqueeze(-1).unsqueeze(-1)  # Reshape to (batch_size, 1024, 1, 1)
        x = F.relu(self.relu_deconv1(x))
        x = F.relu(self.relu_deconv2(x))
        x = F.relu(self.relu_deconv3(x))
        x = torch.sigmoid(self.sigmoid_deconv(x))  # Output image in [0, 1] range
        return x


class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder (VAE) for image reconstruction and latent space learning.

    This model combines an encoder and a decoder with a variational bottleneck to learn a latent
    representation of input images.

    Args:
        latent_dimension (int): Dimension of the latent space. Default is 32.
        image_channels (int): Number of channels in the input images (e.g., 3 for RGB).
          Default is 3.
    """

    def __init__(self, latent_dimension: int = 32, image_channels: int = 3):
        super().__init__()
        self.latent_dimension = latent_dimension
        self.encoder = Encoder(latent_dimension, stride=2)
        self.decoder = Decoder(latent_dimension, image_channels, stride=2)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, image_channels, height, width).

        Returns:
            tuple: A tuple containing:
                - reconstruction (torch.Tensor): Reconstructed images.
                - mu (torch.Tensor): Mean vector of the latent distribution.
                - sigma (torch.Tensor): Standard deviation vector of the latent distribution.
        """
        mu, log_sigma = self.encoder(x)
        sigma = torch.exp(log_sigma)
        z = mu + sigma * torch.randn_like(sigma)
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_sigma

    def get_latent(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Encodes a single observation into the latent space.

        Args:
            observation (torch.Tensor): Input tensor of shape (image_channels, height, width).

        Returns:
            torch.Tensor: Latent vector of shape (latent_dimension,).
        """
        mu, log_sigma = self.encoder(observation)  # Add batch dimension
        sigma = log_sigma.exp()
        return mu + sigma * torch.randn_like(sigma)

    def get_latents(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Encodes a batch of observations into the latent space.

        Args:
            observations (torch.Tensor): Input tensor of shape
                                         (batch_size,ep_len, image_channels, height, width).

        Returns:
            torch.Tensor: Latent vectors of shape (batch_size,ep_len, latent_dimension).
        """
        batch_size, ep_len, *observation_shape = observations.shape
        observations = observations.view(batch_size * ep_len, *observation_shape)
        mu, log_sigma = self.encoder(observations)
        sigma = torch.exp(log_sigma)
        latents = mu + sigma * torch.randn_like(sigma)
        latents = latents.view(batch_size, ep_len, -1)
        return latents

    def loss(
        self,
        reconstruction: torch.Tensor,
        original: torch.Tensor,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the loss for the VAE.

        Args:
            reconstruction (torch.Tensor): Reconstructed images.
            original (torch.Tensor): Original input images.
            mu (torch.Tensor): Mean vector of the latent distribution.
            log_sigma (torch.Tensor): Log standard deviation vector of the latent distribution.

        Returns:
            torch.Tensor: Combined reconstruction and KL-divergence loss.
        """
        # Reconstruction loss (MSE or BCE)
        reconstruction_loss = F.mse_loss(
            input=reconstruction, target=original, reduction="sum"
        )
        # KL-divergence loss
        kl_divergence = -0.5 * torch.sum(
            1 + 2 * log_sigma - mu.pow(2) - (2 * log_sigma).exp()
        )
        return reconstruction_loss + kl_divergence

    @staticmethod
    def from_pretrained(file_path: Path = Path("models") / "vision.pt") -> "ConvVAE":
        """
        Loads a pre-trained ConvVAE model from a file.

        Args:
            file_path (Path): Path to the saved model file.

        Returns:
            ConvVAE: Loaded ConvVAE instance.
        """
        loaded_data = torch.load(file_path, weights_only=False)
        conv_vae = ConvVAE()
        conv_vae.load_state_dict(loaded_data)
        return conv_vae


class VisionTrainer:
    """
    Trainer class for training and evaluating a ConvVAE model on rollout datasets.
    """

    def _train_step(
        self,
        vision: ConvVAE,
        train_dataloader: RolloutDataloader,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Performs a single training step with tqdm progress bar.

        Args:
            vision (ConvVAE): The model to train.
            train_dataloader (RolloutDataloader): DataLoader for the training dataset.
            optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
            epoch (int): Current epoch number.
            total_epochs (int): Total number of epochs.

        Returns:
            float: The average training loss over the dataloader.
        """
        vision.train()
        train_loss = 0

        for batch_episode_observations, _, _ in train_dataloader:
            batch_episode_observations = batch_episode_observations.to(
                next(vision.parameters()).device
            ).permute(1, 0, 2, 3, 4)

            for batch_observations in batch_episode_observations:
                reconstruction, mu, log_sigma = vision(batch_observations)
                loss = vision.loss(reconstruction, batch_observations, mu, log_sigma)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        train_loss /= len(train_dataloader)
        return train_loss

    def _test_step(
        self,
        vision: ConvVAE,
        test_dataloader: RolloutDataloader,
    ) -> float:
        """
        Performs a single evaluation step with tqdm progress bar.

        Args:
            vision (ConvVAE): The model to evaluate.
            test_dataloader (RolloutDataloader): DataLoader for the test dataset.
            epoch (int): Current epoch number.
            total_epochs (int): Total number of epochs.

        Returns:
            float: The average test loss over the dataloader.
        """
        vision.eval()
        test_loss = 0

        for batch_rollouts_observations, _, _ in test_dataloader:
            batch_rollouts_observations = batch_rollouts_observations.to(
                next(vision.parameters()).device
            ).permute(1, 0, 2, 3, 4)

            # "break" the information relative to the sequentiality and shuffle

            shuffled_indices = torch.randperm(batch_rollouts_observations.shape[0])
            batch_rollouts_observations = batch_rollouts_observations[shuffled_indices]
            # Make the resulting shuffled observations a tensor of shape
            # number of elements in an episode x batch_size x observation shape
            batch_rollouts_observations = batch_rollouts_observations.permute(
                1, 0, 2, 3, 4
            )

            for batch_observations in batch_rollouts_observations:
                reconstruction, mu, log_sigma = vision(batch_observations)
                loss = vision.loss(reconstruction, batch_observations, mu, log_sigma)
                test_loss += loss.item()

        test_loss /= len(test_dataloader) * test_dataloader.dataset.get_mean_length()
        return test_loss

    def train(
        self,
        vision: ConvVAE,
        train_dataloader: RolloutDataloader,
        test_dataloader: RolloutDataloader,
        optimizer: torch.optim.Optimizer,
        epochs: int = 10,
        save_path=Path("models") / "vision.pt",
    ):
        """
        Trains the ConvVAE model for a specified number of epochs with tqdm progress bars.

        Args:
            vision (ConvVAE): The model to train.
            train_dataloader (RolloutDataloader): DataLoader for the training dataset.
            test_dataloader (RolloutDataloader): DataLoader for the test dataset.
            optimizer (torch.optim.Optimizer): Optimizer for training the model.
            epochs (int): Number of epochs to train for. Default is 10.
            save_path (Path): Path to save the trained model. Default is 'models/vision.pt'.
        """
        vision.to(next(vision.parameters()).device)

        for epoch in range(epochs):
            train_loss = self._train_step(
                vision,
                train_dataloader,
                optimizer,
            )
            test_loss = 0  # self._test_step(vision, test_dataloader)
            print(
                f"Epoch {epoch + 1}/{epochs}|"
                + f"Train Loss: {train_loss:.4f}|"
                + f"Test Loss: {test_loss:.4f}"
            )

        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(vision.state_dict(), save_path)
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    file_path = Path("data") / "dataset.pt"

    if file_path.exists():
        dataset = RolloutDataset.load(file_path=file_path)
    else:
        dataset = RolloutDataset(num_rollouts=1000, max_steps=400)
        dataset.save(file_path=file_path)

    train_episodes, test_episodes, eval_episodes = torch.utils.data.random_split(
        dataset, [0.5, 0.3, 0.2]
    )

    training_set = RolloutDataset(episodes=train_episodes.dataset.episodes)  # type: ignore
    test_set = RolloutDataset(episodes=test_episodes.dataset.episodes)  # type: ignore
    eval_set = RolloutDataset(episodes=eval_episodes.dataset.episodes)  # type: ignore

    train_dataloader = RolloutDataloader(training_set, 64)
    test_dataloader = RolloutDataloader(test_set, 64)
    vision = ConvVAE().from_pretrained().to(DEVICE)
    # vision = ConvVAE().to(DEVICE)
    # vision_trainer = VisionTrainer()
    # vision_trainer.train(
    #     vision,
    #     train_dataloader,
    #     test_dataloader,
    #     torch.optim.Adam(vision.parameters()),
    #     epochs=20,
    # )
    for i in range(10):
        create_gif(
            dataset.episodes[i + 10],
            vision,
            save_path=Path("vae_reconstructions") / f"reconstruction{i}.gif",
        )
