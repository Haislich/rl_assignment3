"""This module contains the definition for the dataset."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Literal, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
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
        mode: Literal[
            "create",
            "load",
            "from",
        ] = "load",  # Modes: "create", "load", or "from"
        num_rollouts: int = 1,
        max_steps: int = 1,
        continuos: bool = False,
        env_name: str = "CarRacing-v2",
        episodes: Optional[List[Path]] = None,
        root: Path = Path("./data/rollouts"),
    ):
        self.num_rollouts = num_rollouts
        self.max_steps = max_steps
        self.continuos = continuos
        self.env_name = env_name
        self.root = root / ("continuos" if continuos else "discrete")
        self.__transformation = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ]
        )
        if mode == "load":
            self.episodes_paths = self._load_dataset()
            if self.episodes_paths == []:
                print(f"Dataset not found at {self.root}. Falling back to creation.")
                self.episodes_paths = self._create_dataset()
        elif mode == "create":
            self.episodes_paths = self._create_dataset()
            print(f"Created dataset with {len(self.episodes_paths)} episodes.")
        elif mode == "from":
            if episodes is None:
                raise ValueError("'episodes' must be provided when mode is 'from'")
            self.episodes_paths = episodes
        else:
            raise ValueError(
                f"Invalid mode '{mode}'. Supported modes: 'create', 'load', 'from'."
            )
        if self.episodes_paths == []:
            raise ValueError("Cannot create a RolloutDataset without episodes.")

    def _create_dataset(self):
        episodes = self._collect_and_filter_episodes(self.num_rollouts, self.max_steps)
        return episodes

    def _load_dataset(self):
        if not self.root.exists():
            return []
        episode_paths = [
            episode_path for episode_path in sorted(self.root.glob("episode_*"))
        ]
        return episode_paths

    def _sampling_strategy(self, recent_acceleration: bool):
        if self.continuos:
            # Dict continuos actions
            # {
            #     "dtype": dtype("float32"),
            #     "bounded_below": array([True, True, True]),
            #     "bounded_above": array([True, True, True]),
            #     "_shape": (3,),
            #     "low": array([-1.0, 0.0, 0.0], dtype=float32),
            #     "high": array([1.0, 1.0, 1.0], dtype=float32),
            #     "low_repr": "[-1.  0.  0.]",
            #     "high_repr": "1.0",
            #     "_np_random": None,
            # }
            # [steering, acceleration, brake]
            # Idealmente vogliamo che la acc abbia un bias verso l'essere 1
            # mentre la dec abbia un bias verso l'essere 0
            # https://homepage.divms.uiowa.edu/~mbognar/applets/normal.html
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

    def _execute_single_rollout(self, max_steps: int, _index: int = 0) -> Episode:
        observations, actions, rewards = [], [], []
        env = gym.make(
            id=self.env_name,
            continuous=self.continuos,
            # domain_randomize=True
        )

        observation, _ = env.reset()
        recent_acceleration = False
        for _ in range(max_steps):
            action, recent_acceleration = self._sampling_strategy(recent_acceleration)
            next_obs, reward, done, _, _ = env.step(action)
            observation = self.__transformation(observation)
            observations.append(observation)
            action = torch.from_numpy(action)
            actions.append(action)
            reward = torch.tensor(reward)
            rewards.append(reward)
            observation = next_obs
            if done:
                break
        observations = torch.stack(observations)
        actions = torch.stack(actions).to(dtype=torch.float32)
        rewards = torch.stack(rewards).to(dtype=torch.float32)
        return Episode(observations=observations, actions=actions, rewards=rewards)

    def _collect_and_filter_episodes(
        self, num_rollouts: int, max_steps: int
    ) -> List[Path]:
        with ProcessPoolExecutor() as executor:
            data = list(
                tqdm(
                    executor.map(
                        partial(self._execute_single_rollout, max_steps),
                        range(num_rollouts),
                    ),
                    total=num_rollouts,
                    desc="Collecting Rollouts",
                )
            )
        mean_length = int(
            np.sum([len(episode.observations) for episode in data]) / len(data)
        )
        print(f"Mean episode length: {mean_length}")

        filtered_episodes = [
            Episode(
                observations=episode.observations[:mean_length],
                actions=episode.actions[:mean_length],
                rewards=episode.rewards[:mean_length],
            )
            for episode in data
            if len(episode.observations) >= mean_length
        ]
        print(
            "Completed collection and filtering of rollouts.\n"
            + f"{len(filtered_episodes)} episodes retained."
        )

        filtered_episodes_path = [
            episode.save(self.root / f"episode_{idx}.pt")
            for idx, episode in enumerate(filtered_episodes)
        ]
        return filtered_episodes_path

    def __getitem__(self, index):
        episode_path = self.episodes_paths[index]
        return Episode.load(episode_path)

    def __len__(self):
        return len(self.episodes_paths)

    # TODO: Make it coherent w.r.t getitem
    def __iter__(self):
        return iter(self.episodes_paths)


class RolloutDataloader(DataLoader):
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
        batch_observations = torch.stack([episode.observations for episode in batch])
        batch_actions = torch.stack([episode.actions for episode in batch])
        batch_rewards = torch.stack([episode.rewards for episode in batch])

        return batch_observations, batch_actions, batch_rewards

    def __len__(self) -> int:
        return len(self.dataset)  # type:ignore

    def __iter__(
        self,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        yield from super().__iter__()


if __name__ == "__main__":

    dataset = RolloutDataset("create", continuos=False)

#     train_episodes, test_episodes, eval_episodes = torch.utils.data.random_split(
#         dataset, [0.5, 0.3, 0.2]
#     )
#     train_dataset = RolloutDataset(
#         "from", episodes=[dataset.episodes[idx] for idx in train_episodes.indices]
#     )
#     dataloader = RolloutDataloader(train_dataset, batch_size=3)

#     for elem1, elem2, elem3 in dataloader:
#         print(elem1.shape)
