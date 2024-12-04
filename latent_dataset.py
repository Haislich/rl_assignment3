from dataclasses import dataclass
import torch
from vision import ConvVAE
from torch.utils.data import DataLoader, Dataset
from dataset import RolloutDataset, Episode
from pathlib import Path
from typing import List

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class LatentEpisode:
    latent_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor


class LatentDataset(Dataset):
    def __init__(self, episodes: List[Episode], vision: ConvVAE) -> None:
        super().__init__()
        vision_device = next(vision.parameters()).device
        vision.to("cpu")
        latent_episodes = []
        for episode in episodes:
            observations = episode.observations.to("cpu")
            # Ideally I want everything on the CPU and load on demand
            latent_episodes.append(
                LatentEpisode(
                    vision.get_latent(observations),
                    episode.actions,
                    episode.rewards,
                )
            )
            observations.detach()
        vision.to(vision_device)

    def __len__(self) -> int:
        """Returns the number of episodes in the dataset."""
        return len(self.latent_episodes)

    def __getitem__(self, idx: int) -> LatentEpisode:
        """
        Retrieves an episode by index.

        Args:
            idx (int): Index of the episode to retrieve.

        Returns:
            Episode: The episode at the specified index.
        """
        return self.latent_episodes[idx]


if __name__ == "__main__":
    file_path = Path("data") / "dataset.pt"

    if file_path.exists():
        dataset = RolloutDataset.load(file_path=file_path)
    else:
        dataset = RolloutDataset(num_rollouts=1000, max_steps=400)
        dataset.save(file_path=file_path)

    train_episodes, test_episodes, eval_episodes = torch.utils.data.random_split(
        dataset, [0.5, 0.3, 0.2]
    )

    training_set = RolloutDataset(episodes=train_episodes.dataset.episodes)  # type: ignore
    test_set = RolloutDataset(episodes=test_episodes.dataset.episodes)  # type: ignore
    eval_set = RolloutDataset(episodes=eval_episodes.dataset.episodes)  # type: ignore

    vision = ConvVAE().from_pretrained().to(DEVICE)

    train_dataloader = LatentDataset(training_set.episodes, vision)
