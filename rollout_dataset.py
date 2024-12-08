"""This module contains the definition for the dataset."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from multiprocessing import ProcessError
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from tqdm import tqdm


@dataclass
class Episode:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor

    def save(self, episode_path: Path):
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "observations": self.observations,
                "actions": self.actions,
                "rewards": self.rewards,
            },
            episode_path,
        )
        # Not clear if it can go into an exception,
        # This return indicates that everything went fine.
        return episode_path

    @staticmethod
    def load(episode_path: Path):
        if not episode_path.exists():
            raise FileNotFoundError(f"Couldn't find the episode at {episode_path}")
        metadata = torch.load(episode_path, weights_only=True)
        return Episode(
            observations=metadata["observations"],
            actions=metadata["actions"],
            rewards=metadata["rewards"],
        )


class RolloutDataset(Dataset):
    def __init__(
        self,
        num_rollouts: int = 1,
        max_steps: int = 1,
        continuous: bool = True,
        env_name: str = "CarRacing-v2",
        root: Path = Path("./data/rollouts"),
    ):
        self.num_rollouts = num_rollouts
        self.max_steps = max_steps
        self.continuous = continuous
        self.env_name = env_name
        self.root = (
            root / ("continuous" if continuous else "discrete") / f"{max_steps}steps"
        )
        self.__transformation = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ]
        )
        self.episodes_paths = self._load_dataset()

        if len(self.episodes_paths) < num_rollouts:
            self.episodes_paths = self._create_dataset()

        if self.episodes_paths == []:
            raise ValueError("Cannot create a RolloutDataset without episodes.")
        self.episodes_paths = [
            episode_path
            for episode_path in self.episodes_paths
            if episode_path.exists()
        ]
        self._reestablish_episode_ordering()

    @staticmethod
    def from_subset(subset: Subset):
        old_rollout_dataset = subset.dataset
        indices = subset.indices
        new_episodes_paths = np.array(
            old_rollout_dataset.episodes_paths  # type:ignore
        )[indices].tolist()
        new_rollout_dataset = RolloutDataset(
            num_rollouts=old_rollout_dataset.num_rollouts,  # type:ignore
            max_steps=old_rollout_dataset.max_steps,  # type:ignore
            continuous=old_rollout_dataset.continuous,  # type: ignore
            env_name=old_rollout_dataset.env_name,  # type: ignore
        )
        new_rollout_dataset.episodes_paths = new_episodes_paths
        return new_rollout_dataset

    def _reestablish_episode_ordering(self):
        """Reestablish correct episode numbering."""
        if not self.root.exists():
            return

        # Find all episode files and sort them by their numeric suffix
        episode_paths = sorted(
            self.root.glob("episode_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )

        # Rename episodes to have sequential numbering
        for new_index, episode_path in enumerate(episode_paths):
            new_path = self.root / f"episode_{new_index}.pt"
            if episode_path != new_path:
                os.rename(episode_path, new_path)

    def _create_dataset(self):
        episodes_paths = self._collect_and_filter_episodes(
            self.num_rollouts, self.max_steps
        )
        return episodes_paths

    def _load_dataset(self) -> list[Path]:
        if not self.root.exists():
            return []
        episode_paths = [
            episode_path
            for episode_path in sorted(
                self.root.glob("episode_*"), key=lambda p: int(p.stem.split("_")[-1])
            )
        ]
        return episode_paths

    def _sampling_strategy(self, recent_acceleration: bool):
        if self.continuous:
            action = np.random.normal([0, 0.75, 0.25], [0.5, 0.1, 0.1])
            action[0] = action[0].clip(-1, 1)
            action[1] = action[1].clip(0, 1)
            action[2] = action[2].clip(0, 1)
        else:
            if not recent_acceleration:
                action = np.array(3)
                recent_acceleration = True
            else:
                action = np.array(
                    np.random.choice(
                        [0, 1, 2, 3, 4],
                        p=[0.1, 0.3, 0.3, 0.2, 0.1],
                    )
                )
                recent_acceleration = action == 3
        return action, recent_acceleration

    def _execute_single_rollout(self, max_steps: int, index: int = 0) -> Optional[Path]:

        observations, actions, rewards = [], [], []
        env = gym.make(
            id=self.env_name,
            continuous=self.continuous,
        )
        observation, _ = env.reset()
        recent_acceleration = False
        for _ in range(max_steps):
            action, recent_acceleration = self._sampling_strategy(recent_acceleration)
            next_observation, reward, done, _, _ = env.step(action)
            observation = self.__transformation(observation)
            observations.append(observation)
            action = torch.from_numpy(action)
            actions.append(action)
            reward = torch.tensor(reward)
            rewards.append(reward)
            observation = next_observation
            if done:
                break
        observations = torch.stack(observations)
        actions = torch.stack(actions).to(dtype=torch.float32)
        rewards = torch.stack(rewards).to(dtype=torch.float32)
        episode = Episode(observations=observations, actions=actions, rewards=rewards)
        if observations.shape[0] != max_steps:
            return
        try:
            return episode.save(self.root / f"episode_{index}.pt")
        except ProcessError as e:
            print(e)
            return

    def _collect_and_filter_episodes(
        self, num_rollouts: int, max_steps: int
    ) -> List[Path]:
        if len(self.episodes_paths) > num_rollouts:
            return self.episodes_paths
        with ProcessPoolExecutor() as executor:
            episode_paths = list(
                tqdm(
                    executor.map(
                        partial(self._execute_single_rollout, max_steps),
                        range(len(self.episodes_paths), num_rollouts),
                    ),
                    total=num_rollouts - len(self.episodes_paths),
                    desc="Collecting Rollouts",
                )
            )
        episode_paths = [
            episode_path for episode_path in episode_paths if episode_path is not None
        ]
        return self.episodes_paths + episode_paths

    def __getitem__(self, index) -> Path:
        episode_path = self.episodes_paths[index]
        return episode_path

    def __len__(self):
        return len(self.episodes_paths)

    def __iter__(self) -> Iterator[Path]:
        return iter(self.episodes_paths)


class RolloutDataloader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
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
        batch_observations = []
        batch_actions = []
        batch_rewards = []
        for episode_path in batch:
            try:
                episode = Episode.load(episode_path)
                batch_observations.append(episode.observations)
                batch_actions.append(episode.actions)
                batch_rewards.append(episode.rewards)
            except RuntimeError:
                print(f"\t {episode_path.name} seems corrupted, so it will be skipped.")
        batch_observations = torch.stack(batch_observations)
        batch_actions = torch.stack(batch_actions)
        batch_rewards = torch.stack(batch_rewards)

        return batch_observations, batch_actions, batch_rewards

    def __len__(self) -> int:
        return len(self.dataset)  # type:ignore

    def __iter__(
        self,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        yield from super().__iter__()
