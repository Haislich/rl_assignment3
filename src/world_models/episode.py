from dataclasses import dataclass
import torch


@dataclass
class Episode:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor

    def __len__(self):
        return min(
            [
                self.observations.shape[0],
                self.actions.shape[0],
                self.rewards.shape[0],
            ]
        )


@dataclass
class LatentEpisode:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor

    def __len__(self):
        return min(
            [
                self.observations.shape[0],
                self.actions.shape[0],
                self.rewards.shape[0],
            ]
        )
