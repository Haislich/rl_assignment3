"""Vision component"""

from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn
from dataset import RolloutDataloader, Episode
import torchvision.transforms as T
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

    # Trick pylint into hinting
    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(x)


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

    # Trick pylint into hinting
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


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
        mu, log_sigma = self.encoder(observation)
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
    def from_pretrained(model_path: Path = Path("models/vision.pt")) -> "ConvVAE":
        if not model_path.exists():
            raise FileNotFoundError(f"Couldn't find the ConvVae model at {model_path}")
        loaded_data = torch.load(model_path, weights_only=False)
        conv_vae = ConvVAE()
        conv_vae.load_state_dict(loaded_data)
        return conv_vae


class VisionTrainer:
    def _train_step(
        self,
        vision: ConvVAE,
        train_dataloader: RolloutDataloader,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        vision.train()
        train_loss = 0

        for batch_episodes_observations, _, _ in train_dataloader:
            batch_episodes_observations = batch_episodes_observations.to(
                next(vision.parameters()).device
            ).permute(1, 0, 2, 3, 4)
            for batch_observations in batch_episodes_observations:
                reconstruction, mu, log_sigma = vision(batch_observations)
                loss = vision.loss(reconstruction, batch_observations, mu, log_sigma)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_loss /= batch_observations.shape[0]
            train_loss /= batch_episodes_observations.shape[0]
        train_loss /= len(train_dataloader)
        return train_loss

    def _test_step(
        self,
        vision: ConvVAE,
        test_dataloader: RolloutDataloader,
    ) -> float:
        vision.eval()
        test_loss = 0

        for batch_episodes_observations, _, _ in test_dataloader:
            batch_episodes_observations = batch_episodes_observations.to(
                next(vision.parameters()).device
            ).permute(1, 0, 2, 3, 4)
            for batch_observations in batch_episodes_observations:
                reconstruction, mu, log_sigma = vision(batch_observations)
                loss = vision.loss(reconstruction, batch_observations, mu, log_sigma)
                test_loss += loss.item()
                test_loss /= batch_observations.shape[0]
            test_loss /= batch_episodes_observations.shape[0]
        test_loss /= len(test_dataloader)
        return test_loss

    def train(
        self,
        vision: ConvVAE,
        train_dataloader: RolloutDataloader,
        test_dataloader: RolloutDataloader,
        optimizer: torch.optim.Optimizer,
        val_dataloader: Optional[RolloutDataloader] = None,
        epochs: int = 10,
        save_path=Path("models/vision.pt"),
    ):
        vision.to(next(vision.parameters()).device)

        for epoch in range(epochs):
            train_loss = self._train_step(
                vision,
                train_dataloader,
                optimizer,
            )
            test_loss = self._test_step(vision, test_dataloader)
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                + f"Train Loss: {train_loss:.4f} | "
                + f"Test Loss: {test_loss:.4f}"
            )
        if val_dataloader is not None:
            val_loss = self._test_step(vision, test_dataloader)
            print(f"Validation Loss: {val_loss:.4f}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(vision.state_dict(), save_path)
        print(f"Model saved to {save_path}")


def create_gif(
    episode: Episode,
    vision: ConvVAE,
    save_path=Path("vae_reconstruction.gif"),
):
    observations = episode.observations.unsqueeze(0).to(DEVICE)
    latents = vision.get_batched_latents(observations)
    vae_reconstructions = vision.decoder(latents.squeeze(0))
    scale_factor = 1
    spacing = 1
    img_width, img_height = 64 * scale_factor, 64 * scale_factor
    total_width = img_width * 2 + spacing * 2
    total_height = img_height

    images = []
    for t in range(vae_reconstructions.shape[0]):
        original_img = T.Resize((img_height, img_width))(
            T.ToPILImage()(observations[0, t].cpu())
        )
        vae_img = T.Resize((img_height, img_width))(
            T.ToPILImage()(vae_reconstructions[t].cpu())
        )
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
        duration=60,  # Increase duration for slower playback
        loop=0,
    )
    print(f"Vae reconstruction GIF saved to {save_path}")


if __name__ == "__main__":
    from dataset import RolloutDataset, RolloutDataloader

    CONTINUOS = True
    dataset = RolloutDataset(
        "create",
        num_rollouts=20,
        max_steps=20,
        continuos=CONTINUOS,
    )
    (
        train_episodes,
        test_episodes,
        eval_episodes,
    ) = torch.utils.data.random_split(dataset, [0.5, 0.3, 0.2])
    training_set = RolloutDataset(
        "from",
        episodes=[dataset.episodes_paths[idx] for idx in train_episodes.indices],
        continuos=CONTINUOS,
    )
    test_set = RolloutDataset(
        "from",
        episodes=[dataset.episodes_paths[idx] for idx in test_episodes.indices],
        continuos=CONTINUOS,
    )
    eval_set = RolloutDataset(
        "from",
        episodes=[dataset.episodes_paths[idx] for idx in eval_episodes.indices],
        continuos=CONTINUOS,
    )

    train_dataloader = RolloutDataloader(training_set, 64)
    test_dataloader = RolloutDataloader(test_set, 64)
    # vision = ConvVAE().from_pretrained().to(DEVICE)
    vision = ConvVAE().to(DEVICE)
    vision_trainer = VisionTrainer()
    vision_trainer.train(
        vision,
        train_dataloader,
        test_dataloader,
        torch.optim.Adam(vision.parameters()),
        epochs=1,
    )
    for i in range(10):

        create_gif(
            Episode.load(dataset.episodes_paths[i]),
            vision,
            save_path=Path("vae_reconstructions") / f"reconstruction{i}.gif",
        )
