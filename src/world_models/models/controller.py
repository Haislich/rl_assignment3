from pathlib import Path

import numpy as np
import torch
from torch import nn


class Controller(nn.Module):
    def __init__(
        self,
        latent_dimension: int = 32,
        hidden_units: int = 256,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        self.to(device)
        self.device = device
        self.fc = nn.Linear(latent_dimension + hidden_units, 3)

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
        return self

    @staticmethod
    def from_pretrained(
        device,
        model_path: Path = Path("models/controller.pt"),
    ) -> "Controller":
        controller = Controller(device=device)
        if model_path.exists():
            loaded_data = torch.load(model_path, weights_only=False, map_location="cpu")
            controller.load_state_dict(loaded_data["model_state"])
        return controller
