from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from cma import CMAEvolutionStrategy
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from world_models.models.memory import MDN_RNN
from world_models.models.vision import ConvVAE


class Controller(nn.Module):
    def __init__(
        self, latent_dimension: int = 32, hidden_units: int = 256, continuous=True
    ):
        super().__init__()
        self.continuous = continuous
        self.fc = nn.Linear(latent_dimension + hidden_units, 3 if continuous else 1)

    def forward(
        self, latent_observation: torch.Tensor, hidden_state: torch.Tensor
    ) -> torch.Tensor:
        return torch.tanh(
            self.fc(torch.cat((latent_observation, hidden_state), dim=-1))
        )

    def get_weights(self):
        return (
            nn.utils.parameters_to_vector(self.parameters())
            .detach()
            .cpu()
            .numpy()
            .ravel()
        )

    def set_weights(self, weights: np.ndarray):
        nn.utils.vector_to_parameters(
            torch.tensor(weights, dtype=torch.float32), self.parameters()
        )

    @staticmethod
    def from_pretrained(
        model_path: Path = Path("models/controller_continuous.pt"),
    ) -> "Controller":
        if not model_path.exists():
            raise FileNotFoundError(
                f"Couldn't find the  Controller model at {model_path}"
            )
        loaded_data = torch.load(model_path, weights_only=False, map_location="cpu")
        controller = Controller(continuous="continuous" in model_path.name)
        controller.load_state_dict(loaded_data["model_state"])
        return controller
