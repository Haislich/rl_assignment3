import gymnasium as gym

import torch
from torch.utils.data import Dataset, DataLoader


class RolloutDataset(Dataset):
    def __init__(self, env_name, num_rollouts, max_steps):
        self.env = gym.make(env_name)
        self.rollouts = self._collect_rollouts(num_rollouts, max_steps)

    def _collect_rollouts(self, num_rollouts, max_steps):
        data = []
        for _ in range(num_rollouts):
            obs = self.env.reset()
            rollout = []
            for _ in range(max_steps):
                action = self.env.action_space.sample()  # Random action
                next_obs, reward, done, _, _ = self.env.step(action)
                rollout.append((obs, action, reward, next_obs))
                obs = next_obs
                if done:
                    break
            data.append(rollout)
        return data

    def __len__(self):
        return len(self.rollouts)

    def __getitem__(self, idx):
        return self.rollouts[idx]
