from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import torch
from torch import nn
from memory import MDN_RNN
from vision import ConvVAE
from cma import CMAEvolutionStrategy
from multiprocessing import Process, Queue
import gymnasium as gym
from torchvision import transforms
import numpy as np

import copy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Controller(nn.Module):
    def __init__(
        self, latent_dimension: int = 32, hidden_units: int = 256, continuos=True
    ):
        super().__init__()
        self.continuos = continuos
        self.fc = nn.Linear(latent_dimension + hidden_units, 3 if continuos else 1)

    def forward(
        self, latent_observation: torch.Tensor, hidden_state: torch.Tensor
    ) -> torch.Tensor:

        return torch.tanh(
            self.fc(torch.cat((latent_observation, hidden_state), dim=-1))
        )

    def __call__(
        self, latent_observation: torch.Tensor, hidden_state: torch.Tensor
    ) -> torch.Tensor:
        return self.forward(latent_observation, hidden_state)

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


class ControllerTrainer:
    def __init__(
        self,
        controller: Controller,
        vision: ConvVAE,
        memory: MDN_RNN,
        population_size=32,
        env_name="CarRacing-v2",
    ):
        self.controller = controller
        self.vision = vision
        self.memory = memory
        self.population_size = population_size
        self.env_name = env_name
        self.__transformation = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ]
        )

    def _rollout(self, controller: Controller, seed=None):
        env = gym.make(id=self.env_name, continuous=self.memory.continuos)
        if seed is not None:
            observation, _ = env.reset(seed=int(seed))
        else:
            observation, _ = env.reset()
        hidden_state, cell_state = self.memory.init_hidden()
        cumulative_reward = 0

        while True:
            observation: torch.Tensor = self.__transformation(observation)
            latent_observation = self.vision.get_latent(observation.unsqueeze(0))
            latent_observation = latent_observation.unsqueeze(0)
            action = controller.forward(latent_observation, hidden_state)

            numpy_action = action.detach().cpu().numpy().ravel()

            next_observation, reward, done, _, _ = env.step(numpy_action)

            cumulative_reward += float(reward)
            if done:
                break

            print(f"{latent_observation.shape=}")
            print(f"{action.shape=}")
            _mu, _pi, _sigma, hidden_state, cell_state = self.memory.forward(
                latent_observation, action, hidden_state, cell_state
            )
            observation = next_observation
        return cumulative_reward

    def train(self, max_iterations=1):
        initial_solution = self.controller.get_weights()
        print(f"Initial solution size: {len(initial_solution)}")

        solver = CMAEvolutionStrategy(
            initial_solution, 0.2, {"popsize": self.population_size}
        )

        controllers = [Controller() for _ in range(self.population_size)]
        iterations = 0

        while True:
            print(f"Iteration: {iterations}")
            solutions = solver.ask()
            seeds = np.random.randint(0, int(1e6), size=(self.population_size,))
            # fitlist = []
            for controller, solution in zip(controllers, solutions):
                controller.set_weights(solution)
                # total_reward = self._rollout(controller, 3)
                # fitlist.append(total_reward)

            with ProcessPoolExecutor(max_workers=self.population_size) as executor:
                fitlist = list(executor.map(self._rollout, controllers, seeds))

            solver.tell(solutions, fitlist)

            bestsol, bestfit, *_ = solver.result
            print(f"Best fitness in iteration {iterations}: {bestfit}")

            if bestfit > 900:
                print("Task solved!")
                print(f"Best solution: {bestsol}")
                break
            if iterations >= max_iterations:
                print("Task not solved!")
                print(f"Best solution: {bestsol}")
                break

            iterations += 1


vision = ConvVAE.from_pretrained().to("cpu")
memory = MDN_RNN.from_pretrained(Path("models/memory_continuos.pt")).to("cpu")
controller = Controller(continuos=True).to("cpu")
controller_trainer = ControllerTrainer(controller, vision, memory, population_size=2)
controller_trainer.train()
