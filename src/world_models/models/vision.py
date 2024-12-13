"""Vision component"""

from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    class Encoder(nn.Module):
        def __init__(self, latent_dimension: int, *, stride: int = 2):
            super().__init__()
            self.relu_conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=stride)
            self.relu_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=stride)
            self.relu_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=stride)
            self.relu_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=stride)
            self.fc_mu = nn.Linear(2 * 2 * 256, latent_dimension)
            self.fc_sigma = nn.Linear(2 * 2 * 256, latent_dimension)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = F.relu(self.relu_conv1(x))
            x = F.relu(self.relu_conv2(x))
            x = F.relu(self.relu_conv3(x))
            x = F.relu(self.relu_conv4(x))
            x = x.view(x.size(0), -1)
            mu = self.fc_mu(x)
            log_sigma = self.fc_sigma(x)
            return mu, log_sigma

    class Decoder(nn.Module):
        def __init__(
            self, latent_dimension: int, image_channels: int, *, stride: int = 2
        ):
            super().__init__()
            self.fc = nn.Linear(latent_dimension, 1024)
            self.relu_deconv1 = nn.ConvTranspose2d(
                1024, 128, kernel_size=5, stride=stride
            )
            self.relu_deconv2 = nn.ConvTranspose2d(
                128, 64, kernel_size=5, stride=stride
            )
            self.relu_deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=stride)
            self.sigmoid_deconv = nn.ConvTranspose2d(
                32, image_channels, kernel_size=6, stride=stride
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.fc(x)
            x = x.unsqueeze(-1).unsqueeze(-1)
            x = F.relu(self.relu_deconv1(x))
            x = F.relu(self.relu_deconv2(x))
            x = F.relu(self.relu_deconv3(x))
            x = torch.sigmoid(self.sigmoid_deconv(x))
            return x

    def __init__(
        self,
        latent_dimension: int = 32,
        image_channels: int = 3,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        self.to(device)
        self.latent_dimension = latent_dimension
        self.encoder = self.Encoder(latent_dimension, stride=2)
        self.decoder = self.Decoder(latent_dimension, image_channels, stride=2)
        self.device = device

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_sigma = self.encoder(x)
        sigma = torch.exp(log_sigma)
        z = mu + sigma * torch.randn_like(sigma)
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_sigma

    def get_latent(self, observation: torch.Tensor) -> torch.Tensor:
        observation = observation.to(self.device)
        mu, log_sigma = self.encoder.forward(observation)
        sigma = log_sigma.exp()
        return mu + sigma * torch.randn_like(sigma)

    def get_batched_latents(self, batched_observations: torch.Tensor) -> torch.Tensor:
        batch_size, ep_len, *observation_shape = batched_observations.shape
        latents = self.get_latent(
            batched_observations.view(batch_size * ep_len, *observation_shape)
        )
        latents = latents.view(batch_size, ep_len, -1)
        return latents

    def loss(
        self,
        reconstruction: torch.Tensor,
        original: torch.Tensor,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
        _beta: float = 1.0,
    ) -> torch.Tensor:
        reconstruction_loss = (
            F.mse_loss(
                input=reconstruction,
                target=original,
                reduction="sum",
            )
            # / original.numel()
        )
        kl_divergence = -0.5 * torch.sum(
            1 + 2 * log_sigma - mu.pow(2) - (2 * log_sigma).exp()
        )
        return reconstruction_loss + _beta * kl_divergence

    @staticmethod
    def from_pretrained(
        device,
        model_path: Path = Path("models/vision.pt"),
    ) -> "ConvVAE":
        conv_vae = ConvVAE(device=device)
        if model_path.exists():
            loaded_data = torch.load(model_path, weights_only=True, map_location=device)
            conv_vae.load_state_dict(loaded_data["model_state"])
        return conv_vae
