from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from cma import CMAEvolutionStrategy
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from memory import MDN_RNN
from vision import ConvVAE


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


class ControllerTrainer:
    transformation = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )
    vision = ConvVAE.from_pretrained("cpu").eval()
    memory = MDN_RNN.from_pretrained("cpu").eval()

    def __init__(
        self,
        controller: Controller,
        population_size=16,
        env_name="CarRacing-v2",
    ):
        self.controller = controller
        self.env_name = env_name
        self.environment = gym.make(env_name, render_mode="rgb_array")
        self.population_size = population_size

    def _rollout(self, solution, max_steps, index):
        controller = Controller()
        controller.set_weights(solution)
        total_reward = 0
        hidden_state, cell_state = self.memory.init_hidden()

        for _ in tqdm(range(max_steps), desc=f"Worker {index}", leave=False):
            # for _ in range(max_steps):
            episode_reward = 0
            observation, _ = self.environment.reset()
            with torch.no_grad():
                while True:
                    observation: torch.Tensor = self.transformation(observation)
                    latent_observation = self.vision.get_latent(
                        observation.unsqueeze(0)
                    )
                    latent_observation = latent_observation.unsqueeze(0)
                    action = controller(latent_observation, hidden_state)
                    numpy_action = action.detach().cpu().numpy().ravel()
                    next_observation, reward, terminated, truncated, _ = (
                        self.environment.step(numpy_action)
                    )
                    episode_reward += float(reward)
                    done = terminated or truncated
                    if done:
                        break
                    _mu, _pi, _sigma, hidden_state, cell_state = self.memory.forward(
                        latent_observation,
                        action,
                        hidden_state,
                        cell_state,
                    )
                    observation = next_observation
            total_reward += episode_reward
        ret = total_reward / max_steps
        return ret

    def train(
        self,
        max_epochs=1,
        max_steps=10,
        save_path: Path = Path("models/controller_continuous.pt"),
    ):
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize CMA-ES
        initial_solution = self.controller.get_weights()
        bestfit = float("-inf")
        if save_path.exists():
            controller_metadata = torch.load(save_path, weights_only=True)
            initial_solution = controller_metadata["model_state"]
            bestfit = controller_metadata["bestfit"]
        solver = CMAEvolutionStrategy(
            initial_solution, 0.5, {"popsize": self.population_size}
        )
        progress_bar = tqdm(
            range(max_epochs),
            total=max_epochs,
            desc="Calculating solutions with CMAES",
            leave=False,
        )

        for _ in progress_bar:
            solutions = solver.ask()
            fitlist = []
            for index, solution in enumerate(solutions):
                fitlist.append(-self._rollout(solution, max_steps, index))

            solver.tell(solutions, fitlist)
            epoch_bestsol, epoch_bestfit, *_ = solver.result
            epoch_bestfit = -epoch_bestfit
            tqdm.write(f"{epoch_bestfit=}")
            if epoch_bestfit > bestfit:
                bestfit = epoch_bestfit
                self.controller.set_weights(epoch_bestsol)
                torch.save(
                    {
                        "model_state": self.controller.state_dict(),
                        "bestfit": epoch_bestfit,
                    },
                    save_path,
                )
