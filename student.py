from pathlib import Path
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

from controller import Controller, ControllerTrainer
from latent_dataset import LatentDataloader, LatentDataset
from memory import MDN_RNN, MemoryTrainer
from rollout_dataset import RolloutDataloader, RolloutDataset
from vision import ConvVAE, VisionTrainer


class Policy(nn.Module):

    def __init__(self, device=torch.device("cpu"), continuous=True):
        super(Policy, self).__init__()
        self.device = device
        self.continuous = continuous
        self.vision = ConvVAE().to(device)
        self.memory = MDN_RNN().to(device)
        self.controller = Controller().to("cpu")
        self._hidden_state, self._cell_state = self.memory.init_hidden()
        self.transformation = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((64, 64)),
                T.ToTensor(),
            ]
        )

    def forward(self, x):
        # TODO
        return x

    # def act(self, observation: np.ndarray, vision: ConvVAE, memory: MDN_RNN):
    #     observation = transforms.Compose(
    #
    #     )(observation)
    #     latent_observation = self.vision.get_latent(observation.unsqueeze(0))
    #     latent_observation = latent_observation.unsqueeze(0)
    #     action = self.controller(latent_observation, hidden_state)
    #     numpy_action = action.detach().cpu().numpy().ravel()
    def act(self, state):
        observation: torch.Tensor = self.transformation(state)
        latent_observation = self.vision.get_latent(observation.unsqueeze(0))
        latent_observation = latent_observation.unsqueeze(0)
        action = self.controller(latent_observation, self._hidden_state)
        _mu, _pi, _sigma, self._hidden_state, self._cell_state = self.memory.forward(
            latent_observation,
            action,
            self._hidden_state,
            self._cell_state,
        )
        return action.detach().cpu().numpy().ravel()

    def train(
        self,
        num_rollouts=10000,
        max_steps=1000,
        batch_size=128,
        vision_epochs=10,
        memory_epochs=10,
        controller_epochs=10,
        population_size=16,
    ):
        rollout_dataset = RolloutDataset(
            num_rollouts, max_steps, continuous=self.continuous
        )

        (
            train_episodes,
            test_episodes,
            val_episodes,
        ) = torch.utils.data.random_split(rollout_dataset, [0.5, 0.3, 0.2])
        train_dataset = RolloutDataset.from_subset(train_episodes)
        test_dataset = RolloutDataset.from_subset(test_episodes)
        val_dataset = RolloutDataset.from_subset(val_episodes)
        train_dataloader = RolloutDataloader(train_dataset, batch_size)
        test_dataloader = RolloutDataloader(test_dataset, batch_size)
        val_dataloader = RolloutDataloader(val_dataset, batch_size)

        vision_trainer = VisionTrainer(self.vision)
        vision_trainer.train(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            val_dataloader=val_dataloader,
            epochs=vision_epochs,
            optimizer=torch.optim.Adam(self.vision.parameters()),
        )

        latent_training_set = LatentDataset(train_dataset, self.vision)
        latent_test_set = LatentDataset(test_dataset, self.vision)
        latent_val_set = LatentDataset(val_dataset, self.vision)

        train_dataloader = LatentDataloader(latent_training_set, batch_size)
        test_dataloader = LatentDataloader(latent_test_set, batch_size)
        val_dataloader = LatentDataloader(latent_val_set, batch_size)

        memory_trainer = MemoryTrainer(self.memory)
        memory_trainer.train(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            val_dataloader=val_dataloader,
            optimizer=torch.optim.Adam(self.memory.parameters()),
            epochs=memory_epochs,
        )

        controller_trainer = ControllerTrainer(
            self.controller, self.vision, self.memory, population_size=population_size
        )
        controller_trainer.train(controller_epochs, max_steps)
        return

    def save(self):
        pass
        # torch.save(self.state_dict(), "model.pt")

    def load(self):
        try:
            self.vision = ConvVAE().from_pretrained().to(self.device)
            self.memory = MDN_RNN().from_pretrained().to(self.device)
            self.controller = Controller().from_pretrained().to("cpu")
        except FileNotFoundError:
            self.vision = ConvVAE().to(self.device)
            self.memory = MDN_RNN().to(self.device)
            self.controller = Controller().to("cpu")
        # self.load_state_dict(torch.load("model.pt", map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = Policy(device)
    policy.train(
        num_rollouts=100,
        max_steps=30,
        batch_size=32,
        vision_epochs=5,
        memory_epochs=5,
        controller_epochs=10,
        population_size=16,
    )
    # policy.train()
