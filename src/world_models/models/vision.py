"""Vision component"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Tuple
import torch
import torch.nn.functional as F
from torch import nn

from world_models.dataset import Episode


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

    def __init__(self, latent_dimension: int = 32, image_channels: int = 3):
        super().__init__()
        self.latent_dimension = latent_dimension
        self.encoder = self.Encoder(latent_dimension, stride=2)
        self.decoder = self.Decoder(latent_dimension, image_channels, stride=2)

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


@dataclass
class LatentEpisode:
    latent_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor

    @staticmethod
    def from_episode_path(episode_path: Path, vision: ConvVAE):
        latent_episode = None
        try:
            episode = Episode.load(episode_path)
            device = next(vision.parameters()).device
            latent_observations = vision.get_latent(episode.observations.to(device))
            latent_episode = LatentEpisode(
                latent_observations=latent_observations,
                actions=episode.actions,
                rewards=episode.rewards,
            )
        except FileNotFoundError:
            print(f"Episode {episode_path} hasn't been found, it will be skipped")

        return latent_episode

    def save(self, latent_episode_path: Path):
        latent_episode_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "latent_observations": self.latent_observations,
                "actions": self.actions,
                "rewards": self.rewards,
            },
            latent_episode_path,
        )
        return latent_episode_path

    @staticmethod
    def load(latent_episode_path: Path):
        if not latent_episode_path.exists():
            raise FileNotFoundError(
                f"Couldn't find the episode at {latent_episode_path}"
            )
        metadata = torch.load(
            latent_episode_path, weights_only=True, map_location="cpu"
        )
        return LatentEpisode(
            latent_observations=metadata["latent_observations"],
            actions=metadata["actions"],
            rewards=metadata["rewards"],
        )


class LatentDataset(Dataset):
    def __init__(
        self,
        rollout_dataset: RolloutDataset,
        vision: ConvVAE,
        root: Path = Path("./data/latents"),
    ):
        self.rollout_dataset = rollout_dataset
        if len(self.rollout_dataset) == 0:
            raise ValueError("Rollout Dataset must be non empty.")
        self.vision = vision.eval()
        self.root = (
            root
            / ("continuous" if rollout_dataset.continuous else "discrete")
            / f"{rollout_dataset.max_steps}steps"
        )
        self.latent_episodes_paths = self._load_dataset()
        self.latent_episodes_paths = self._create_dataset()
        if self.latent_episodes_paths == []:
            raise ValueError("The latent dataset cannot be empty")

    def _create_dataset(self):
        self.root.mkdir(parents=True, exist_ok=True)
        total_indices = {
            int(path.stem.split("_")[-1])
            for path in self.rollout_dataset.episodes_paths
        }
        current_indices = {
            int(path.stem.split("_")[-1]) for path in self.latent_episodes_paths
        }
        missing_indices = total_indices - current_indices
        missing_episodes = []
        for index in missing_indices:
            missing_episode_path = self.rollout_dataset.root / f"episode_{index}.pt"
            missing_latent_episode_path = self.root / f"latent_episode_{index}.pt"
            if missing_episode_path.exists():
                missing_episodes.append(
                    (missing_episode_path, missing_latent_episode_path)
                )
        latent_episodes_paths = []
        if missing_episodes:
            for missing_episode_path, missing_latent_episode_path in tqdm(
                missing_episodes,
                total=len(missing_episodes),
                desc="Generating latent_dataset",
            ):
                latent_episode = LatentEpisode.from_episode_path(
                    missing_episode_path, self.vision
                )
                if latent_episode is not None:
                    latent_episodes_paths.append(
                        latent_episode.save(missing_latent_episode_path)
                    )

        return latent_episodes_paths + self.latent_episodes_paths

    def _load_dataset(self) -> List[Path]:
        if not self.root.exists():
            return []
        latent_episodes_paths = []
        for episode_path in self.rollout_dataset.episodes_paths:
            latent_episode_path = self.root / f"latent_{episode_path.name}"
            if latent_episode_path.exists():
                latent_episodes_paths.append(latent_episode_path)

        return latent_episodes_paths

    def __len__(self):
        return len(self.latent_episodes_paths)

    def __getitem__(self, index) -> Path:
        latent_episode_path = self.latent_episodes_paths[index]
        return latent_episode_path

    def __iter__(self):
        return iter(self.latent_episodes_paths)


class LatentDataloader(DataLoader):
    def __init__(
        self,
        dataset: LatentDataset,
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        sampler: Sampler | Iterable | None = None,
        batch_sampler: Sampler[List] | Iterable[List] | None = None,
        num_workers: int = 0,
        collate_fn: Callable[[List], Any] | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Callable[[int], None] | None = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
    ):
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            self.__collate_fn,
            pin_memory,
            drop_last,
            timeout,
            worker_init_fn,
            multiprocessing_context,
            generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )

    @staticmethod
    def __collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_latent_observations = []
        batch_actions = []
        batch_rewards = []
        for latent_episode_path in batch:
            try:
                latent_episode = LatentEpisode.load(latent_episode_path)
                batch_latent_observations.append(latent_episode.latent_observations)
                batch_actions.append(latent_episode.actions)
                batch_rewards.append(latent_episode.rewards)
            except RuntimeError:
                print(
                    f"\t {latent_episode_path.name} seems corrupted, so it will be skipped."
                )
        batch_latent_observations = torch.stack(batch_latent_observations)
        batch_actions = torch.stack(batch_actions)
        batch_rewards = torch.stack(batch_rewards)

        return batch_latent_observations, batch_actions, batch_rewards

    def __len__(self) -> int:
        return len(self.dataset)  # type:ignore

    def __iter__(
        self,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        yield from super().__iter__()
