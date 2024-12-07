"""Vision component"""

from pathlib import Path
from typing import Optional
from math import ceil

import torch
import torch.nn.functional as F

# import torchvision.transforms as T
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from rollout_dataset import RolloutDataloader


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
        x = self.fc(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.relu_deconv1(x))
        x = F.relu(self.relu_deconv2(x))
        x = F.relu(self.relu_deconv3(x))
        x = torch.sigmoid(self.sigmoid_deconv(x))
        return x


class ConvVAE(nn.Module):
    def __init__(self, latent_dimension: int = 32, image_channels: int = 3):
        super().__init__()
        self.latent_dimension = latent_dimension
        self.encoder = Encoder(latent_dimension, stride=2)
        self.decoder = Decoder(latent_dimension, image_channels, stride=2)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(x)
        mu, log_sigma = self.encoder(x)
        sigma = torch.exp(log_sigma)
        z = mu + sigma * torch.randn_like(sigma)
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_sigma

    def get_latent(self, observation: torch.Tensor) -> torch.Tensor:
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
    ) -> torch.Tensor:
        reconstruction_loss = F.mse_loss(
            input=reconstruction,
            target=original,
            reduction="sum",
        )
        kl_divergence = -0.5 * torch.sum(
            1 + 2 * log_sigma - mu.pow(2) - (2 * log_sigma).exp()
        )
        return reconstruction_loss + kl_divergence

    @staticmethod
    def from_pretrained(
        model_path: Path = Path("models/vision/vision.pt"),
    ) -> "ConvVAE":
        if not model_path.exists():
            raise FileNotFoundError(f"Couldn't find the ConvVae model at {model_path}")
        loaded_data = torch.load(model_path, weights_only=False)
        conv_vae = ConvVAE()
        conv_vae.load_state_dict(loaded_data)
        return conv_vae


class VisionTrainer:

    def __init__(self, vision: ConvVAE) -> None:
        self.vision = vision
        self.device = next(self.vision.parameters()).device

    def _train_step(
        self,
        train_dataloader: RolloutDataloader,
        optimizer: torch.optim.Optimizer,
        save_path: Path,
        epoch: int,
        resume_batch_idx: int = 0,
    ) -> float:
        self.vision.train()
        train_loss = 0

        # Create epoch folder
        epoch_save_path = save_path / f"epoch_{epoch}"
        epoch_save_path.mkdir(parents=True, exist_ok=True)

        for batch_idx, (batch_episodes_observations, _, _) in enumerate(
            tqdm(
                train_dataloader,
                total=ceil(
                    len(train_dataloader) / train_dataloader.batch_size  # type:ignore
                ),
                desc="Loading episode batches from the dataloader",
                leave=False,
            )
        ):
            if batch_idx < resume_batch_idx:
                continue

            episodes_batch_observations = batch_episodes_observations.to(
                self.device
            ).permute(1, 0, 2, 3, 4)

            for batch_observations in tqdm(
                episodes_batch_observations,
                total=episodes_batch_observations.shape[0],
                desc="Processing timesteps for current episode batch",
                leave=False,
            ):
                reconstruction, mu, log_sigma = self.vision.forward(batch_observations)
                loss = self.vision.loss(
                    reconstruction, batch_observations, mu, log_sigma
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= batch_episodes_observations.shape[0]

            # Save batch checkpoint
            batch_checkpoint_path = epoch_save_path / f"batch_{batch_idx}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "model_state": self.vision.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                batch_checkpoint_path,
            )

        train_loss /= len(train_dataloader)
        return train_loss

    def _test_step(
        self,
        test_dataloader: RolloutDataloader,
    ) -> float:
        self.vision.eval()
        test_loss = 0
        for batch_episodes_observations, _, _ in tqdm(
            test_dataloader,
            total=ceil(
                len(test_dataloader) / test_dataloader.batch_size  # type:ignore
            ),
            desc="Loading episode batches from the dataloader",
            leave=False,
        ):
            episodes_batch_observations = batch_episodes_observations.to(
                self.device
            ).permute(1, 0, 2, 3, 4)
            for batch_observations in tqdm(
                episodes_batch_observations,
                total=episodes_batch_observations.shape[0],
                desc="Testing timesteps for current episode batch",
                leave=False,
            ):
                reconstruction, mu, log_sigma = self.vision.forward(batch_observations)
                loss = self.vision.loss(
                    reconstruction, batch_observations, mu, log_sigma
                )
                test_loss += loss.item()
            test_loss /= batch_episodes_observations.shape[0]
        test_loss /= len(test_dataloader)
        return test_loss

    def train(
        self,
        train_dataloader: RolloutDataloader,
        test_dataloader: RolloutDataloader,
        optimizer: torch.optim.Optimizer,
        val_dataloader: Optional[RolloutDataloader] = None,
        epochs: int = 10,
        save_path=Path("models"),
        log_dir=Path("logs/vision"),
    ):
        save_path = save_path / "vision"
        save_path.mkdir(parents=True, exist_ok=True)

        # Initialize SummaryWriter for TensorBoard
        writer = SummaryWriter(log_dir=log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Load checkpoint if available
        resume_epoch = 0
        resume_batch_idx = 0
        checkpoint_path = sorted(
            save_path.glob("epoch_*/batch_*.pt"),
            key=lambda p: (
                int(p.parent.stem.split("_")[1]),  # Extract epoch number
                int(p.stem.split("_")[1]),  # Extract batch number
            ),
        )
        if len(checkpoint_path) > 0:
            last_checkpoint = checkpoint_path[-1]
            checkpoint = torch.load(last_checkpoint, weights_only=True)
            self.vision.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            resume_epoch = checkpoint["epoch"]
            resume_batch_idx = checkpoint["batch_idx"] + 1

        for epoch in tqdm(
            range(resume_epoch, epochs), total=epochs, desc="Training Vision"
        ):
            # Training Step
            train_loss = self._train_step(
                train_dataloader,
                optimizer,
                save_path,
                epoch,
                resume_batch_idx if epoch == resume_epoch else 0,
            )
            # Testing Step
            test_loss = self._test_step(test_dataloader)

            # Log to TensorBoard
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Test", test_loss, epoch)
            print()
            print(
                f"\tEpoch {epoch + 1}/{epochs} | "
                + f"Train Loss: {train_loss:.4f} | "
                + f"Test Loss: {test_loss:.4f}"
            )

            resume_batch_idx = 0  # Reset after resuming the first epoch

        if val_dataloader is not None:
            val_loss = self._test_step(test_dataloader)
            print(f"Validation Loss: {val_loss:.4f}")
        # Save final model
        torch.save(self.vision.state_dict(), save_path / "vision.pt")
        print(f"Model saved to {save_path}")

        # Close SummaryWriter
        writer.close()
