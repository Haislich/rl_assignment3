from dataclasses import dataclass
from typing import Generator
import gymnasium as gym
from PIL import Image
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


@dataclass
class Rollout:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor


class RolloutDataset(Dataset):
    def __init__(
        self,
        num_rollouts=10000,
        max_steps=100,  # TODO: Remove
        continuous=False,
        env_name="CarRacing-v2",
    ):
        # TODO: Handle the discrete case
        self.env = gym.make(id=env_name, continuous=continuous)  # ,render_mode="human")
        self.max_steps = max_steps
        self.__transformation = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.rollouts = self._collect_rollouts(num_rollouts, max_steps)
        self._filter_and_trucate_rollouts()

    def _collect_rollouts(self, num_rollouts, max_steps):
        data = []
        for _ in range(num_rollouts):
            observations = []
            actions = []
            rewards = []
            obs, _ = self.env.reset()
            for _ in range(max_steps):
                action = self.env.action_space.sample()  # Random action
                next_obs, reward, done, _, _ = self.env.step(action)
                obs = transforms.ToTensor()(
                    transforms.Resize((64, 64))(Image.fromarray(obs))
                ).permute(0, 1, 2)
                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                obs = next_obs
                # Simulate the lack of max_steps
                # if np.random.rand() < 0.90:
                #     break
                if done:
                    break

            observations = torch.stack(observations)
            rewards = torch.Tensor(rewards)
            actions = torch.Tensor(actions)
            data.append(
                Rollout(observations=observations, actions=actions, rewards=rewards)
            )
        return data

    def _filter_and_trucate_rollouts(
        self,
    ) -> None:
        lengths = [len(rollout.observations) for rollout in self.rollouts]
        mean_length = int(torch.tensor(lengths, dtype=torch.float32).mean().item())

        # Filter and truncate rollouts
        filtered_rollouts = []
        for rollout in self.rollouts:
            if len(rollout.observations) >= mean_length:
                # Truncate observations, actions, and rewards to the mean length
                truncated_observations = rollout.observations[:mean_length]
                truncated_actions = rollout.actions[:mean_length]
                truncated_rewards = rollout.rewards[:mean_length]
                filtered_rollouts.append(
                    Rollout(
                        observations=truncated_observations,
                        actions=truncated_actions,
                        rewards=truncated_rewards,
                    )
                )
        self.rollouts = filtered_rollouts

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
        self.dataset = dataset
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.__collate_fn,
        )

    def __collate_fn(self, batch):
        # for elem in batch:
        #     print(elem.observations.shape)
        # exit()
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
