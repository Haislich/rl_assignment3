"""This module contains the definition for the dataset."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Literal, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

from dataset import Episode, RolloutDataset
from vision import ConvVAE


@dataclass
class LatentEpisode:
    latent_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor

    @staticmethod
    def from_episode(episode: Episode, vision: ConvVAE):
        latent_observations = vision.get_latent(episode.observations)
        return LatentEpisode(
            latent_observations=latent_observations,
            actions=episode.actions,
            rewards=episode.rewards,
        )

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
        # Not clear if it can go into an exception,
        # This return indicates that everything went fine.
        return latent_episode_path

    @staticmethod
    def load(latent_episode_path: Path):
        if not latent_episode_path.exists():
            raise FileNotFoundError(
                f"Couldn't find the episode at {latent_episode_path}"
            )
        metadata = torch.load(latent_episode_path, weights_only=True)
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
        mode: Literal[
            "create",
            "load",
        ] = "load",  # Modes: "create", "load", or "from"
        root: Path = Path("./data/latents"),
    ):
        self.rollout_dataset = rollout_dataset
        if len(self.rollout_dataset) == 0:
            raise ValueError("Rollout Dataset must be non empty.")
        self.vision = vision
        self.root = root / ("continuos" if rollout_dataset.continuos else "discrete")
        if mode == "create":
            self.latent_episodes_paths = self._create_dataset()
        elif mode == "load":
            self.latent_episodes_paths = self._load_dataset()
            if self.latent_episodes_paths == []:
                print(
                    f"Latent Dataset not found at {self.root}. Falling back to creation."
                )
                self.latent_episodes_paths = self._create_dataset()
        else:
            raise ValueError(
                f"Invalid mode '{mode}'. Supported modes: 'create', 'load'."
            )

    def _create_dataset(self):
        return [
            LatentEpisode.from_episode(Episode.load(episode_path), self.vision).save(
                self.root / f"latent_episode_{idx}.pt",
            )
            for idx, episode_path in enumerate(self.rollout_dataset)
        ]

    def _load_dataset(self):
        if not self.root.exists():
            return []
        latent_episode_paths = [
            latent_episode_path
            for latent_episode_path in sorted(self.root.glob("latent_episode_*"))
        ]
        return latent_episode_paths

    def __len__(self):
        return len(self.latent_episodes_paths)

    def __getitem__(self, index):
        latent_episode_path = self.latent_episodes_paths[index]
        return LatentEpisode.load(latent_episode_path)

    def __iter__(self):
        return iter(self.latent_episodes_paths)


class LatentDataloader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
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
        batch_latent_observations = torch.stack(
            [latent_episode.latent_observations for latent_episode in batch]
        )
        batch_actions = torch.stack(
            [latent_episode.actions for latent_episode in batch]
        )
        batch_rewards = torch.stack(
            [latent_episode.rewards for latent_episode in batch]
        )

        return batch_latent_observations, batch_actions, batch_rewards

    def __len__(self) -> int:
        return len(self.dataset)  # type:ignore

    def __iter__(
        self,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        yield from super().__iter__()


# if __name__ == "__main__":
#     rollout_dataset = RolloutDataset("load")
#     vision = ConvVAE.from_pretrained()
#     latent_dataset = LatentDataset(rollout_dataset, vision, "create")
#     for latent_episode_path in latent_dataset:
#         print(LatentEpisode.load(latent_episode_path).latent_observations.shape)
