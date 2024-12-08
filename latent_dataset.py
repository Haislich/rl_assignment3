"""This module contains the definition for the dataset."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
from rollout_dataset import Episode, RolloutDataset

from vision import ConvVAE


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
        except Exception as e:  # type:ignore
            print(
                f"Some unknown exception has happend with {episode_path}, it will be skipped"
            )
            print(f"\tException:{e}")

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
