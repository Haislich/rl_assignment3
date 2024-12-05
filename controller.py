from pathlib import Path
import torch
from torch import nn
from memory import MDN_RNN
from vision import ConvVAE
from cma import CMAEvolutionStrategy
from multiprocessing import Process, Queue
import gymnasium as gym

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Controller(nn.Module):
    def __init__(
        self, latent_dimension: int = 32, hidden_units: int = 256, continuos=False
    ):
        super().__init__()
        self.continuos = continuos
        self.fc = nn.Linear(latent_dimension + hidden_units, 3 if continuos else 1)

    def forward(self, latent: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        print(latent.device)
        print(hidden.device)

        return torch.tanh(self.fc(torch.cat((latent, hidden), dim=-1)))


memory = MDN_RNN.from_pretrained(Path("models/memory_continuos.pt")).to(DEVICE)

l = torch.Tensor(1, 32).to(DEVICE)
a = torch.Tensor(1, 3).to(DEVICE)
_, _, _, h = memory.forward(l, a)

print(Controller().to(DEVICE)(l, h))

# class ControllerTrainer:
#     def __init__(
#         self,controller,vision,memory,env_name = "CarRacing-v2"
#     ) -> None:
#         self.controller = controller
#         self.vision = vision
#         self.memory = memory
#         self.env_name = env_name
#         self.solution = Queue(1)

#     def _rollout(self,  vision, memory,controller,):
#         env = gym.make(
#             id=self.env_name,
#             continuous=self.memory.continuos,
#         )
#         observation, _ =env.reset()
#         while not done:
#             z = vision.get_latent(obs)
#             a = controller([z, h])
#             obs, reward, done = env.step(a)
#             cumulative_reward += reward
#             h = rnn.forward([a, z, h])
#         return cumulative_reward
#     def fit(
#         self,

#     ):
#         solver = CMAEvolutionStrategy()
#         while True:
