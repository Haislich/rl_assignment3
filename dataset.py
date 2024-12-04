"""This module contains the definition for the dataset."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import imageio
from pathlib import Path


@dataclass
class Episode:
    """
    Represents a single episode in a reinforcement learning task.

    An episode is a sequence of transitions, containing observations, actions, and rewards.
    The lengths of these tensors correspond to the number of steps in the episode (`ep_len`).

    Attributes:
        observations (torch.Tensor): Tensor of shape (ep_len, 3, 64, 64) representing
            the sequence of observations, where each observation is a 64x64 RGB image.
        actions (torch.Tensor): Tensor of shape (ep_len,) representing the sequence of actions
            taken by the agent during the episode.
        rewards (torch.Tensor): Tensor of shape (ep_len,) representing the sequence of rewards
            received for each step in the episode.
    """

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor


class RolloutDataset(Dataset):
    """
    Dataset for storing and managing episodes in a reinforcement learning environment.

    This class collects episodes from a specified Gym environment, processes the observations,
    and stores them for training or evaluation purposes. Episodes can also be saved and loaded.

    Args:
        num_rollouts (int): Number of episodes to collect if none are provided. Default is 10,000.
        max_steps (int): Maximum number of steps per episode. Default is 100.
        continuous (bool): Whether the environment uses continuous action space. Default is False.
        env_name (str): Name of the Gym environment. Default is "CarRacing-v2".
        episodes (Optional[List[Episode]]): Pre-collected episodes to initialize the dataset.
    """

    def __init__(
        self,
        num_rollouts: int = 10000,
        max_steps: int = 100,
        reward_threshold: float = -0.1,
        continuos: bool = False,
        env_name: str = "CarRacing-v2",
        episodes: Optional[List["Episode"]] = None,
    ):
        self.num_rollouts = num_rollouts
        self.env_name = env_name
        self.continuos = continuos
        self.reward_threshold = reward_threshold
        self.seeds = np.random.randint(0, 10000, size=num_rollouts)
        self.__transformation = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ]
        )
        if episodes is not None:
            self.episodes = episodes
            self.max_steps = self.get_mean_length()
        else:
            self.max_steps = max_steps
            self.episodes = self._collect_and_filter_rollouts(num_rollouts)

    def _collect_single_rollout(self, index: int = 0) -> "Episode":
        """
        Collects a single episode by interacting with the environment.

        Args:
            index (int): Index of the rollout for tracking.

        Returns:
            Episode: A single episode containing observations, actions, and rewards.
        """
        observations, actions, rewards = [], [], []
        env = gym.make(
            id=self.env_name,
            continuous=self.continuos,  # domain_randomize=True
        )

        observation, _ = env.reset()
        # observation, _ = env.reset(options={"random_map": True})

        recent_acceleration = False
        for _ in range(self.max_steps):
            # action = np.where(
            #     np.random.uniform(0, 1) < 0.3, 3, self.env.action_space.sample()
            # ).item()
            if not recent_acceleration:
                # Ensure acceleration is selected if it hasn't occurred recently
                action = 3
                recent_acceleration = True
            else:
                # Randomly select other actions with a bias towards valid actions
                action = np.random.choice(
                    [0, 1, 2, 3, 4],
                    p=[0.1, 0.3, 0.3, 0.2, 0.1],
                ).item()
                recent_acceleration = action == 3

            next_obs, reward, done, _, _ = env.step(action)
            observation = self.__transformation(observation)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            observation = next_obs
            if done:
                break

        # Convert to torch tensors
        observations = torch.stack(observations)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        return Episode(observations=observations, actions=actions, rewards=rewards)

    def _collect_and_filter_rollouts(self, num_rollouts: int) -> List["Episode"]:
        """
        Collects multiple episodes in parallel and filters them to a consistent length.

        Args:
            num_rollouts (int): Number of episodes to collect.

        Returns:
            List[Episode]: List of processed episodes.
        """
        print(f"Starting collection of {num_rollouts} rollouts...")

        # Use tqdm to track overall progress

        with ProcessPoolExecutor() as executor:
            data = list(
                tqdm(
                    executor.map(
                        self._collect_single_rollout,
                        range(num_rollouts),
                    ),
                    total=num_rollouts,
                    desc="Collecting Rollouts",
                )
            )
        thresholded_data = []
        for episode in data:
            rewards_mean = episode.rewards.mean().item()
            if rewards_mean >= self.reward_threshold:
                thresholded_data.append(episode)

        assert len(thresholded_data) > 0, "No episode has the minimum desired reward"
        # Calculate the mean length of episodes
        mean_length = int(
            torch.tensor(
                [len(episode.observations) for episode in thresholded_data],
                dtype=torch.float32,
            )
            .mean()
            .item()
        )
        print(f"Mean episode length: {mean_length}")

        # Filter and truncate episodes
        filtered_episodes = [
            Episode(
                observations=episode.observations[:mean_length],
                actions=episode.actions[:mean_length],
                rewards=episode.rewards[:mean_length],
            )
            for episode in thresholded_data
            if len(episode.observations) >= mean_length
        ]

        print(
            "Completed collection and filtering of rollouts.\n"
            + f"{len(filtered_episodes)} episodes retained."
        )
        return filtered_episodes

    def get_mean_length(self) -> int:
        """
        Computes the mean length of episodes in the dataset.

        Returns:
            int: Mean episode length.
        """
        return int(
            torch.tensor(
                [len(episode.observations) for episode in self.episodes],
                dtype=torch.float32,
            )
            .mean()
            .item()
        )

    def save(self, file_path: Path):
        """
        Saves the dataset to a file.

        Args:
            file_path (Path): Path to save the dataset file.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data_to_save = {
            "episodes": [
                {
                    "observations": episode.observations,
                    "actions": episode.actions,
                    "rewards": episode.rewards,
                }
                for episode in self.episodes
            ],
        }
        torch.save(data_to_save, file_path)
        print(f"Dataset saved to {file_path}")

    @classmethod
    def load(cls, file_path: Path) -> "RolloutDataset":
        """
        Loads a dataset from a file.

        Args:
            file_path (Path): Path to the dataset file.

        Returns:
            RolloutDataset: An instance of the dataset loaded with episodes.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        loaded_data = torch.load(file_path, weights_only=False)
        episodes = [
            Episode(
                observations=data["observations"],
                actions=data["actions"],
                rewards=data["rewards"],
            )
            for data in loaded_data["episodes"]
        ]
        return cls(num_rollouts=0, max_steps=0, episodes=episodes)

    def __len__(self) -> int:
        """Returns the number of episodes in the dataset."""
        return len(self.episodes)

    def __getitem__(self, idx: int) -> "Episode":
        """
        Retrieves an episode by index.

        Args:
            idx (int): Index of the episode to retrieve.

        Returns:
            Episode: The episode at the specified index.
        """
        return self.episodes[idx]

    def create_gif(self, episode: Episode, save_path: Path):
        """
        Creates a GIF for a specific episode's observations.

        Args:
            episode (Episode): The episode to visualize.
            save_path (Path): Path to save the GIF file.
        """
        # Ensure save directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert tensor observations to images
        frames = []
        for frame_tensor in episode.observations:
            # Convert tensor to NumPy array for imageio
            frame = (
                frame_tensor.permute(1, 2, 0).numpy() * 255
            )  # (C, H, W) -> (H, W, C)
            frame = frame.astype(np.uint8)
            frames.append(frame)

        # Save the frames as a GIF
        imageio.mimsave(save_path, frames, fps=10)
        print(f"GIF saved to {save_path}")


class RolloutDataloader(DataLoader):
    """
    A custom DataLoader for the RolloutDataset, designed to handle episodes of rollouts in
    reinforcement learning environments.

    This class batches episodes into tensors of observations, actions, and rewards for
    efficient processing in machine learning pipelines.

    Args:
        dataset (RolloutDataset): The dataset containing episodes to be loaded.
        batch_size (int): The number of episodes in each batch. Default is 32.
        shuffle (bool): Whether to shuffle the dataset before each epoch. Default is True.
    """

    def __init__(
        self,
        dataset: RolloutDataset,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        self.dataset: RolloutDataset = dataset

        self.batch_size: int = batch_size
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.__collate_fn,
        )

    @staticmethod
    def __collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Custom collation function for batching episodes.

        Args:
            batch (List[Episode]): A list of episodes to collate into a batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - Tensor of observations of shape (batch_size, ep_len, channels, height, width).
                - Tensor of actions of shape (batch_size, ep_len).
                - Tensor of rewards of shape (batch_size, ep_len).
        """
        batch_observations = torch.stack([rollout.observations for rollout in batch])
        batch_actions = torch.stack([rollout.actions for rollout in batch])
        batch_rewards = torch.stack([rollout.rewards for rollout in batch])

        return batch_observations, batch_actions, batch_rewards

    def __len__(self) -> int:
        """
        Returns the total number of steps in the dataset, calculated as the
        product of the number of episodes and the mean episode length.

        Returns:
            int: Total number of steps in the dataset.
        """
        return len(self.dataset)  # * self.dataset.get_mean_length()

    def __iter__(
        self,
    ) -> Iterator[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:  # -> Generator[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Custom iterator for the DataLoader that yields batches of episodes.

        Yields:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - Tensor of observations of shape (batch_size, ep_len, channels, height, width).
                - Tensor of actions of shape (batch_size, ep_len).
                - Tensor of rewards of shape (batch_size, ep_len).
        """
        yield from super().__iter__()


# if __name__ == "__main__":
#     file_path = Path("data") / "dataset.pt"

#     if False and file_path.exists():
#         dataset = RolloutDataset.load(file_path=file_path)
#     else:
#         dataset = RolloutDataset(num_rollouts=10, max_steps=1000, reward_threshold=-0.1)

#         dataset.save(file_path=file_path)
#     # Select an episode
#     # selected_episode = dataset[0]
#     for i in range(5):
#         # Save GIF
#         save_path = Path("episodes") / f"episode_{i}.gif"
#         dataset.create_gif(dataset[i], save_path)
