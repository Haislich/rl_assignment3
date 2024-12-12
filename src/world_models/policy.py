from pathlib import Path
import torch
import torchvision.transforms as T
from world_models.models.controller import Controller
from world_models.models.memory import MDN_RNN
from world_models.models.vision import ConvVAE
import numpy as np
from world_models.dataset import RolloutDataset


class Policy:
    transformation = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor(),
        ]
    )

    def __init__(
        self,
    ):
        super().__init__()
        try:
            self.vision = ConvVAE.from_pretrained("cpu")
            self.memory = MDN_RNN.from_pretrained("cpu")
            self.controller = Controller.from_pretrained()
        except FileNotFoundError:
            self.vision = ConvVAE().to("cpu")
            self.memory = MDN_RNN().to("cpu")
            self.controller = Controller().to("cpu")
        self.vision = self.vision.eval()
        self.memory = self.memory.eval()
        self.controller = self.controller.eval()
        self._hidden_state, self._cell_state = self.memory.init_hidden()

    def act(self, state) -> np.ndarray:
        with torch.no_grad():
            observation: torch.Tensor = self.transformation(state)
            latent_observation = self.vision.get_latent(observation.unsqueeze(0))
            latent_observation = latent_observation.unsqueeze(0)
            action = self.controller(latent_observation, self._hidden_state)
            _mu, _pi, _sigma, self._hidden_state, self._cell_state = (
                self.memory.forward(
                    latent_observation,
                    action,
                    self._hidden_state,
                    self._cell_state,
                )
            )
            return action.detach().cpu().numpy().ravel()


class PolicyDataset(RolloutDataset):
    def __init__(
        self,
        num_rollouts: int,
        max_steps: int,
        *,
        env_name: str = "CarRacing-v3",
        root: Path = Path("./data/rollouts"),
    ):
        super().__init__(num_rollouts, max_steps, env_name=env_name, root=root)
        self.policy = Policy()

    def sampling_strategy(self, _env, observation: np.ndarray) -> np.ndarray:
        return self.policy.act(observation)
