from dataclasses import dataclass
import gymnasium as gym
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os


@dataclass
class Rollout:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor


class RolloutDataset(Dataset):
    def __init__(
        self,
        num_rollouts=10000,
        max_steps=100,
        continuous=False,
        env_name="CarRacing-v2",
        rollouts=None,
    ):
        self.env = gym.make(id=env_name, continuous=continuous)
        self.__transformation = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ]
        )
        if rollouts is not None:
            self.rollouts = rollouts
        else:
            self.rollouts = self._collect_and_filter_rollouts(num_rollouts, max_steps)

    def _collect_and_filter_rollouts(self, num_rollouts, max_steps):
        """Collects rollouts, filters and truncates them to a mean length."""
        data = []
        for rollout in range(num_rollouts):
            print(f"Rollout {rollout}")
            observations, actions, rewards = [], [], []
            observation, _ = self.env.reset()
            for _ in range(max_steps):
                action = self.env.action_space.sample()
                next_obs, reward, done, _, _ = self.env.step(action)
                observation = self.__transformation(Image.fromarray(observation))
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

            data.append(
                Rollout(observations=observations, actions=actions, rewards=rewards)
            )

        mean_length = int(
            torch.tensor(
                [len(rollout.observations) for rollout in data], dtype=torch.float32
            )
            .mean()
            .item()
        )

        # Filter and truncate rollouts
        filtered_rollouts = []
        for rollout in data:
            if len(rollout.observations) >= mean_length:
                filtered_rollouts.append(
                    Rollout(
                        observations=rollout.observations[:mean_length],
                        actions=rollout.actions[:mean_length],
                        rewards=rollout.rewards[:mean_length],
                    )
                )

        return filtered_rollouts

    def get_mean_length(self):
        return int(
            torch.tensor(
                [len(rollout.observations) for rollout in self.rollouts],
                dtype=torch.float32,
            )
            .mean()
            .item()
        )

    def save(self, file_path: Path):
        # Serialize rollouts and mean_length to a file
        os.makedirs(file_path.parents[0], exist_ok=True)
        data_to_save = {
            "rollouts": [
                {
                    "observations": rollout.observations,
                    "actions": rollout.actions,
                    "rewards": rollout.rewards,
                }
                for rollout in self.rollouts
            ],
        }
        torch.save(data_to_save, file_path)
        print(f"Dataset saved to {file_path}")

    @classmethod
    def load(cls, file_path):
        # Load rollouts and mean_length from a file
        loaded_data = torch.load(file_path, weights_only=True)
        rollouts = [
            Rollout(
                observations=data["observations"],
                actions=data["actions"],
                rewards=data["rewards"],
            )
            for data in loaded_data["rollouts"]
        ]
        # Create an instance with the loaded data
        instance = cls(num_rollouts=0, max_steps=0, rollouts=rollouts)
        return instance

    def __len__(self):
        return len(self.rollouts)

    def __getitem__(self, idx) -> Rollout:
        return self.rollouts[idx]


class RolloutDataloader(DataLoader):
    def __init__(
        self,
        dataset: RolloutDataset,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        self.dataset: RolloutDataset = dataset
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.__collate_fn,
        )

    def __collate_fn(self, batch):
        batch_rollouts_observations = torch.stack(
            [rollout.observations for rollout in batch]
        )
        batch_rollouts_actions = torch.stack([rollout.actions for rollout in batch])
        batch_rollouts_rewards = torch.stack([rollout.rewards for rollout in batch])

        return (
            batch_rollouts_observations,
            batch_rollouts_actions,
            batch_rollouts_rewards,
        )

    def __len__(self) -> int:
        return len(self.dataset) * self.dataset.get_mean_length()

    def __iter__(self):
        for (
            batch_rollouts_observations,
            batch_rollouts_actions,
            batch_rollouts_rewards,
        ) in super().__iter__():

            yield (
                batch_rollouts_observations,
                batch_rollouts_actions,
                batch_rollouts_rewards,
            )
