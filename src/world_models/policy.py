import torch
import torchvision.transforms as T
from world_models.models.controller import Controller
from world_models.models.memory import MDN_RNN
from world_models.models.vision import ConvVAE
import numpy as np


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
        vision: ConvVAE,
        memory: MDN_RNN,
        controller: Controller,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__()
        self.device = device
        self.vision = vision.to(device)
        self.memory = memory.to(device)
        self.controller = controller.to(device)
        # self.vision = ConvVAE.from_pretrained(device)
        # self.memory = MDN_RNN.from_pretrained(device)
        # self.controller = Controller.from_pretrained(device)
        self._hidden_state, self._cell_state = self.memory.init_hidden()

    def act(self, state) -> np.ndarray:
        self.vision = self.vision.eval()
        self.memory = self.memory.eval()
        self.controller = self.controller.eval()
        with torch.no_grad():
            observation: torch.Tensor = self.transformation(state).to(
                self.device
            )  # type : ignore
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

    def evolve(self, weights: np.ndarray):
        self.controller.set_weights(weights)
