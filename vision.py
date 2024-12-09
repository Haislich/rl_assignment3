"""Vision component"""

from pathlib import Path
from typing import Optional
from math import ceil

import torch
import torch.nn.functional as F
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
        _beta: float = 0.1,
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
        if not model_path.exists():
            raise FileNotFoundError(f"Couldn't find the ConvVae model at {model_path}")
        loaded_data = torch.load(model_path, weights_only=False, map_location=device)
        conv_vae = ConvVAE()
        conv_vae.load_state_dict(loaded_data["model_state"])
        return conv_vae


class VisionTrainer:
    def __init__(self, vision: ConvVAE) -> None:
        self.vision = vision
        self.device = next(self.vision.parameters()).device

    def _train_step(
        self,
        train_dataloader: RolloutDataloader,
        optimizer: torch.optim.Optimizer,
        beta: float = 0.1,
    ) -> float:
        self.vision.train()
        train_loss = 0

        for batch_episodes_observations, _, _ in tqdm(
            train_dataloader,
            total=ceil(
                len(train_dataloader) / train_dataloader.batch_size  # type:ignore
            ),
            desc="Loading episode batches from the dataloader",
            leave=False,
        ):

            batch_size, seq_len, *obs_shape = batch_episodes_observations.shape

            # Step 1: Reshape the tensor to (batch_size * seq_len, *obs_shape)
            compacted_observations = batch_episodes_observations.view(-1, *obs_shape)

            # Step 2: Shuffle along the first dimension
            shuffled_indices = torch.randperm(compacted_observations.size(0))
            shuffled_observations = compacted_observations[shuffled_indices]

            # Step 3: Reshape back to (seq_len, batch_size, *obs_shape)
            shuffled_episodes_batched_observations = shuffled_observations.view(
                seq_len, batch_size, *obs_shape
            ).to(self.device)
            for batched_observations in tqdm(
                shuffled_episodes_batched_observations,
                total=shuffled_episodes_batched_observations.shape[0],
                desc="Processing timesteps for current episode batch",
                leave=False,
            ):
                reconstruction, mu, log_sigma = self.vision.forward(
                    batched_observations
                )
                loss = self.vision.loss(
                    reconstruction, batched_observations, mu, log_sigma, beta
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= batch_episodes_observations.shape[0]
        train_loss /= len(train_dataloader)
        return train_loss

    def _test_step(
        self, test_dataloader: RolloutDataloader, beta: float = 0.1
    ) -> float:
        self.vision.eval()
        test_loss = 0
        with torch.inference_mode():
            for batch_episodes_observations, _, _ in tqdm(
                test_dataloader,
                total=ceil(
                    len(test_dataloader) / test_dataloader.batch_size  # type:ignore
                ),
                desc="Loading episode batches from the dataloader",
                leave=False,
            ):
                batch_size, seq_len, *obs_shape = batch_episodes_observations.shape

                # Step 1: Reshape the tensor to (batch_size * seq_len, *obs_shape)
                compacted_observations = batch_episodes_observations.view(
                    -1, *obs_shape
                )

                # Step 2: Shuffle along the first dimension
                shuffled_indices = torch.randperm(compacted_observations.size(0))
                shuffled_observations = compacted_observations[shuffled_indices]

                # Step 3: Reshape back to (seq_len, batch_size, *obs_shape)
                shuffled_episodes_batched_observations = shuffled_observations.view(
                    seq_len, batch_size, *obs_shape
                ).to(self.device)
                for batched_observations in tqdm(
                    shuffled_episodes_batched_observations,
                    total=shuffled_episodes_batched_observations.shape[0],
                    desc="Testing timesteps for current episode batch",
                    leave=False,
                ):
                    reconstruction, mu, log_sigma = self.vision.forward(
                        batched_observations
                    )
                    loss = self.vision.loss(
                        reconstruction, batched_observations, mu, log_sigma, beta
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
        save_path=Path("models/vision.pt"),
        log_dir=Path("logs/vision"),
    ):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Initialize SummaryWriter for TensorBoard
        writer = SummaryWriter(log_dir=log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        initial_epoch = 0
        if save_path.exists():
            vision_metadata = torch.load(
                save_path, weights_only=True, map_location=self.device
            )
            initial_epoch = vision_metadata["epoch"]
            self.vision.load_state_dict(vision_metadata["model_state"])
            optimizer.load_state_dict(vision_metadata["optimizer_state"])
        for epoch in tqdm(
            range(initial_epoch, epochs + initial_epoch),
            total=epochs,
            desc="Training Vision",
            leave=False,
        ):
            beta = min(1.0, epoch / epochs)
            train_loss = self._train_step(
                train_dataloader,
                optimizer,
                beta,
            )
            # Testing Step
            test_loss = 0  # self._test_step(test_dataloader, beta)

            # Log to TensorBoard
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Test", test_loss, epoch)

            tqdm.write(
                f"\tEpoch {epoch + 1}/{epochs+initial_epoch} | "
                f"Train Loss: {train_loss:.5f} | "
                f"Test Loss: {test_loss:.5f}"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": self.vision.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                save_path,
            )

        # if val_dataloader is not None:
        #     val_loss = self._test_step(test_dataloader)
        #     print(f"Validation Loss: {val_loss:.4f}")
        print(f"Model saved to {save_path}")

        # Close SummaryWriter
        writer.close()


# if __name__ == "__main__":
#     from rollout_dataset import RolloutDataset, RolloutDataloader, Episode
#     from PIL import Image

#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#     num_rollouts = 10000
#     max_steps = 1000
#     continuous = True
#     batch_size = 32
#     rollout_dataset = RolloutDataset(num_rollouts, max_steps, continuous=continuous)

#     (
#         train_episodes,
#         test_episodes,
#         val_episodes,
#     ) = torch.utils.data.random_split(rollout_dataset, [0.5, 0.3, 0.2])
#     train_dataset = RolloutDataset.from_subset(train_episodes)
#     test_dataset = RolloutDataset.from_subset(test_episodes)
#     val_dataset = RolloutDataset.from_subset(val_episodes)
#     train_dataloader = RolloutDataloader(train_dataset, batch_size)
#     test_dataloader = RolloutDataloader(test_dataset, batch_size)
#     val_dataloader = RolloutDataloader(val_dataset, batch_size)
#     vision = ConvVAE().to(DEVICE)
#     vision_trainer = VisionTrainer(vision)
#     vision_trainer.train(
#         train_dataloader=train_dataloader,
#         test_dataloader=test_dataloader,
#         val_dataloader=val_dataloader,
#         epochs=1,
#         optimizer=torch.optim.Adam(vision.parameters()),
#     )

#     def create_vision_gif(
#         episode: Episode,
#         vision: ConvVAE,
#         save_path=Path("media/vision_reconstruction.gif"),
#     ):
#         observations = episode.observations.unsqueeze(0).to(DEVICE)
#         latents = vision.get_batched_latents(observations)
#         vae_reconstructions = vision.decoder(latents.squeeze(0))
#         scale_factor = 1
#         spacing = 1
#         img_width, img_height = 64 * scale_factor, 64 * scale_factor
#         total_width = img_width * 2 + spacing * 2
#         total_height = img_height

#         images = []
#         for t in range(vae_reconstructions.shape[0]):
#             original_img = T.Resize((img_height, img_width))(
#                 T.ToPILImage()(observations[0, t].cpu())
#             )
#             vae_img = T.Resize((img_height, img_width))(
#                 T.ToPILImage()(vae_reconstructions[t].cpu())
#             )
#             combined_img = Image.new("RGB", (total_width, total_height), (0, 0, 0))
#             combined_img.paste(original_img, (0, 0))
#             combined_img.paste(vae_img, (img_width + spacing, 0))
#             images.append(combined_img)

#         save_path.parent.mkdir(parents=True, exist_ok=True)
#         # Save as GIF
#         images[0].save(
#             save_path,
#             save_all=True,
#             append_images=images[1:],
#             duration=len(images) / 60,
#             loop=0,
#         )
#         print(f"Vision reconstruction GIF saved to {save_path}")

#     episode = Episode.load(rollout_dataset[0])
#     create_vision_gif(
#         episode,
#         vision,
#     )
