"""This module contains the definition for the dataset."""

from typing import Any, Callable, Iterable, Iterator, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from world_models.episode import Episode, LatentEpisode


class RolloutDataset(Dataset[Episode | LatentEpisode]):
    def __init__(self, episodes: List[Episode] | List[LatentEpisode]):
        self.num_rollouts = len(episodes)
        self.steps = min(len(episode) for episode in episodes)
        self.episodes = []
        for episode in episodes:
            episode.observations = episode.observations[: self.steps]
            episode.actions = episode.actions[: self.steps]
            episode.rewards = episode.rewards[: self.steps]
            self.episodes.append(episode)

    def __getitem__(self, index) -> Episode | LatentEpisode:
        return self.episodes[index]

    def __len__(self) -> int:
        return len(self.episodes)

    def __iter__(self) -> Iterator[Episode | LatentEpisode]:
        return iter(self.episodes)


class RolloutDataloader(DataLoader):
    def __init__(
        self,
        dataset: RolloutDataset,
        batch_size: int = 1,
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
        observations = []
        actions = []
        rewards = []
        for episode in batch:
            observations.append(episode.observations)
            actions.append(episode.actions)
            rewards.append(episode.rewards)
        batch_observations: torch.Tensor = torch.stack(observations)
        batch_actions = torch.stack(actions)
        batch_rewards = torch.stack(rewards)

        return batch_observations, batch_actions, batch_rewards
