"""This module contains the definition for the dataset."""

from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
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
        return episode_path

    @staticmethod
    def load(episode_path: Path):
        if not episode_path.exists():
            raise FileNotFoundError(f"Couldn't find the episode at {episode_path}")
        metadata = torch.load(episode_path, weights_only=True, map_location="cpu")
        return Episode(
            observations=metadata["observations"],
            actions=metadata["actions"],
            rewards=metadata["rewards"],
        )


class RolloutDataset(ABC, Dataset[Path]):
    class RolloutDatasetSubset(Subset):
        dataset: "RolloutDataset"

    transformation = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )
    episodes_paths: List[Path]

    def __init__(
        self,
        num_rollouts: int,
        max_steps: int,
        *,
        env_name: str = "CarRacing-v3",
        root: Path = Path("./data/rollouts"),
    ):
        self.num_rollouts = num_rollouts
        self.max_steps = max_steps
        self.env_name = env_name
        self.root = root / f"{max_steps}_steps"
        self.episodes_paths = self._load_dataset()
        if len(self.episodes_paths) < num_rollouts:
            self.episodes_paths = self._create_dataset()
        self.episodes_paths = [
            episode_path
            for episode_path in self.episodes_paths
            if episode_path.exists()
        ]
        if self.episodes_paths == []:
            raise ValueError(f"Cannot create a {type(self).__name__} without episodes.")

    @staticmethod
    @abstractmethod
    def from_subset(subset: RolloutDatasetSubset) -> "RolloutDataset":
        pass

    def _load_dataset(self) -> list[Path]:
        if not self.root.exists():
            return []
        episode_paths = [
            episode_path
            for episode_path in sorted(
                self.root.glob("episode_*"), key=lambda p: int(p.stem.split("_")[-1])
            )
        ]
        return episode_paths[: min(self.num_rollouts, len(episode_paths))]

    @abstractmethod
    def sampling_strategy(
        self, environment: gym.Env, observation: np.ndarray
    ) -> np.ndarray: ...

    def _execute_single_rollout(self, index: int = 0) -> Optional[Path]:
        episode_path = None
        observations, actions, rewards = [], [], []
        env = gym.make(id=self.env_name, continuous=True)
        observation, _ = env.reset()
        for _ in range(self.max_steps):
            action = self.sampling_strategy(env, observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            observation = self.transformation(observation)
            observations.append(observation)
            action = torch.from_numpy(action)
            actions.append(action)
            reward = torch.tensor(reward)
            rewards.append(reward)
            observation = next_observation
            done = terminated or truncated
            if done:
                break
        observations = torch.stack(observations)
        actions = torch.stack(actions).to(dtype=torch.float32)
        rewards = torch.stack(rewards).to(dtype=torch.float32)
        episode = Episode(observations=observations, actions=actions, rewards=rewards)
        if observations.shape[0] == self.max_steps:
            episode_path = episode.save(self.root / f"episode_{index}.pt")
        return episode_path

    def _create_dataset(
        self,
    ) -> List[Path]:
        if len(self.episodes_paths) > self.num_rollouts:
            return self.episodes_paths
        with ProcessPoolExecutor() as executor:
            episode_paths = list(
                tqdm(
                    executor.map(
                        self._execute_single_rollout,
                        range(len(self), self.num_rollouts),
                    ),
                    total=self.num_rollouts - len(self),
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

    def __len__(self) -> int:
        return len(self.episodes_paths)

    def __iter__(self) -> Iterator[Path]:
        return iter(self.episodes_paths)


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
        for episode_path in batch:
            try:
                episode = Episode.load(episode_path)
                observations.append(episode.observations)
                actions.append(episode.actions)
                rewards.append(episode.rewards)
            except RuntimeError:
                print(f"\t {episode_path.name} seems corrupted, so it will be skipped.")
        batch_observations: torch.Tensor = torch.stack(observations)
        batch_actions = torch.stack(actions)
        batch_rewards = torch.stack(rewards)

        return batch_observations, batch_actions, batch_rewards

    def __iter__(
        self,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        yield from super().__iter__()


class RandomDataset(RolloutDataset):
    def __init__(
        self,
        num_rollouts: int,
        max_steps: int,
        *,
        env_name: str = "CarRacing-v3",
        root: Path = Path("./data/rollouts"),
    ):
        super().__init__(num_rollouts, max_steps, env_name=env_name, root=root)

    def sampling_strategy(self, _env: gym.Env, _observation: np.ndarray) -> np.ndarray:
        action = np.random.normal([0, 0.75, 0.75], [0.25, 0.1, 0.1])
        action[0] = np.clip(action[0], -1, 1)
        action[1] = np.clip(action[1], 0, 1)
        action[2] = np.clip(action[2], 0, 1)
        return action

    @staticmethod
    def from_subset(subset: RolloutDataset.RolloutDatasetSubset):
        old_rollout_dataset: RolloutDataset = subset.dataset
        indices = subset.indices
        new_episodes_paths = np.array(old_rollout_dataset.episodes_paths)[
            indices
        ].tolist()
        new_rollout_dataset = RandomDataset(
            num_rollouts=old_rollout_dataset.num_rollouts,
            max_steps=old_rollout_dataset.max_steps,
            env_name=old_rollout_dataset.env_name,
        )
        new_rollout_dataset.episodes_paths = new_episodes_paths
        return new_rollout_dataset


if __name__ == "__main__":
    rollout_dataset = RandomDataset(101, max_steps=1)
    rollout_dataloader = RolloutDataloader(rollout_dataset, batch_size=3)
    print(len(rollout_dataloader))
