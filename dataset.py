from dataclasses import dataclass
import gymnasium as gym
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset


@dataclass
class Rollout:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor


class RolloutDataset(Dataset):
    def __init__(
        self,
        num_rollouts,
        max_steps=10000,
        continuous=False,
        env_name="CarRacing-v2",
    ):
        # TODO: Handle the discrete case
        self.env = gym.make(id=env_name, continuous=continuous)  # ,render_mode="human")
        self.transformation = transforms.Compose(
            [
                transforms.Resize((64, 64)),  # Resize to 64x64
                transforms.ToTensor(),  # Convert to a PyTorch tensor
            ]
        )
        self.rollouts = self._collect_rollouts(num_rollouts, max_steps)

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
                )
                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                obs = next_obs
                if done:
                    break
            observations = torch.Tensor(observations)
            rewards = torch.Tensor(rewards)
            actions = torch.Tensor(actions)
            data.append(
                Rollout(observations=observations, actions=actions, rewards=rewards)
            )
        return data

    def __len__(self):
        return len(self.rollouts)

    def __getitem__(self, idx):
        return self.rollouts[idx]


if __name__ == "__main__":
    dataset = RolloutDataset(num_rollouts=1, max_steps=10)
