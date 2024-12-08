import copy
import math

# from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from cma import CMAEvolutionStrategy
from matplotlib.animation import FuncAnimation
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
        loaded_data = torch.load(model_path, weights_only=True)
        controller = Controller(continuous="continuous" in model_path.name)
        controller.load_state_dict(loaded_data["model_state"])
        return controller


class ControllerTrainer:
    def __init__(
        self,
        controller: Controller,
        vision: ConvVAE,
        memory: MDN_RNN,
        population_size=16,
        env_name="CarRacing-v2",
        render=False,
    ):
        self.controller = controller
        self.vision = vision.to("cpu").eval()
        self.memory = memory.to("cpu").eval()
        self.population_size = population_size
        self.env_name = env_name
        self.n_rows, self.n_cols = self._get_rows_and_cols()
        self.controllers = [
            copy.deepcopy(controller) for _ in range(self.population_size)
        ]
        self.render = render
        self.__transformation = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ]
        )

    def _get_rows_and_cols(self):
        sqrt_num = math.sqrt(self.population_size)
        n_rows = math.floor(sqrt_num)
        n_cols = math.ceil(self.population_size / n_rows)
        while n_rows * n_cols < self.population_size:
            n_rows += 1
            n_cols = math.ceil(self.population_size / n_rows)
        return n_rows, n_cols

    def _rollout(self, args):
        worker, max_steps = args
        environment = gym.make(self.env_name, render_mode="rgb_array")
        observation, _ = environment.reset()
        hidden_state, cell_state = self.memory.init_hidden()
        cumulative_reward = 0
        frames = []
        progress_bar = tqdm(
            range(max_steps), desc=f"Worker {worker}: Reward {cumulative_reward:.2f}"
        )
        for _ in progress_bar:
            frames.append(observation)
            observation: torch.Tensor = self.__transformation(observation)
            latent_observation = self.vision.get_latent(observation.unsqueeze(0))
            latent_observation = latent_observation.unsqueeze(0)
            action = self.controller(latent_observation, hidden_state)
            numpy_action = action.detach().cpu().numpy().ravel()
            next_observation, reward, done, _, _ = environment.step(numpy_action)
            cumulative_reward += float(reward)
            if done:
                break
            _mu, _pi, _sigma, hidden_state, cell_state = self.memory.forward(
                latent_observation,
                action,
                hidden_state,
                cell_state,
            )
            observation = next_observation
            progress_bar.set_description(
                f"Worker {worker}: Reward {cumulative_reward:.2f}"
            )
        environment.close()
        return frames, cumulative_reward

    def animate_rollouts(self, all_frames):
        """Animate the collected frames after all rollouts are complete."""
        fig, ax = plt.subplots(self.n_rows, self.n_cols, figsize=(10, 8))
        ax = ax.flatten()
        images = []
        for axis in ax:
            img = axis.imshow(np.zeros((64, 64, 3), dtype=np.uint8), vmin=0, vmax=255)
            axis.axis("off")
            images.append(img)

        def update(frame):
            for i, img in enumerate(images):
                if i < len(all_frames) and frame < len(
                    all_frames[i]
                ):  # Ensure valid indices
                    img.set_data(all_frames[i][frame])
            return images

        max_frames = max(len(frames) for frames in all_frames)
        _anim = FuncAnimation(fig, update, frames=max_frames, interval=50, blit=True)
        plt.show()

    def train(
        self,
        max_epochs=1,
        max_steps=10,
        save_path: Path = Path("models/controller_continuous.pt"),
    ):

        save_path.parent.mkdir(parents=True, exist_ok=True)
        initial_epoch = 0
        if save_path.exists():
            controller_metadata = torch.load(save_path, weights_only=True)
            initial_epoch = controller_metadata["epoch"]
            self.controller.load_state_dict(controller_metadata["model_state"])

        initial_solution = self.controller.get_weights()
        solver = CMAEvolutionStrategy(
            initial_solution, 0.2, {"popsize": self.population_size}
        )
        for epoch in tqdm(
            range(initial_epoch, max_epochs + initial_epoch),
            total=max_epochs,
            desc="Calculating solutions with CMAES",
            leave=False,
        ):
            solutions = solver.ask()
            with ThreadPoolExecutor(max_workers=self.population_size) as executor:
                results = list(
                    executor.map(
                        self._rollout,
                        zip(
                            range(self.population_size),
                            [max_steps] * self.population_size,
                        ),
                    )
                )

            if self.render:
                all_frames = [result[0] for result in results]
                self.animate_rollouts(all_frames)
            fitlist = [result[1] for result in results]
            solver.tell(solutions, fitlist)
            bestsol, bestfit, *_ = solver.result
            print()
            print(f"\tBest fitness in epoch {epoch}: {bestfit}")
            if bestfit > 900:
                print("\tBest solution found")
                break
            self.controller.set_weights(bestsol)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": self.controller.state_dict(),
                },
                save_path,
            )
        print(f"Model saved to {save_path}")
