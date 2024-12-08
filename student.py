from pathlib import Path
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image
from controller import Controller, ControllerTrainer
from latent_dataset import LatentDataloader, LatentDataset
from memory import MDN_RNN, MemoryTrainer
from rollout_dataset import Episode, RolloutDataloader, RolloutDataset
from vision import ConvVAE, VisionTrainer


def create_dataset_gif(
    episode: Episode,
    device,
    save_path=Path("media/rollout_dataset.gif"),
):
    observations = episode.observations.unsqueeze(0).to(device)
    scale_factor = 1
    img_width, img_height = 64 * scale_factor, 64 * scale_factor
    total_width = img_width
    total_height = img_height
    images = []
    for t in range(observations.shape[1]):
        original_img = T.Resize((img_height, img_width))(
            T.ToPILImage()(observations[0, t].cpu())
        )
        combined_img = Image.new("RGB", (total_width, total_height), (0, 0, 0))
        combined_img.paste(original_img, (0, 0))
        images.append(combined_img)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Save as GIF
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=len(images) / 60,
        loop=0,
    )
    print(f"Dataset GIF saved to {save_path}")


def create_vision_gif(
    episode: Episode,
    vision: ConvVAE,
    device,
    save_path=Path("media/vision_reconstruction.gif"),
):
    observations = episode.observations.unsqueeze(0).to(device)
    latents = vision.get_batched_latents(observations)
    vae_reconstructions = vision.decoder(latents.squeeze(0))
    scale_factor = 1
    spacing = 1
    img_width, img_height = 64 * scale_factor, 64 * scale_factor
    total_width = img_width * 2 + spacing * 2
    total_height = img_height

    images = []
    for t in range(vae_reconstructions.shape[0]):
        original_img = T.Resize((img_height, img_width))(
            T.ToPILImage()(observations[0, t].cpu())
        )
        vae_img = T.Resize((img_height, img_width))(
            T.ToPILImage()(vae_reconstructions[t].cpu())
        )
        combined_img = Image.new("RGB", (total_width, total_height), (0, 0, 0))
        combined_img.paste(original_img, (0, 0))
        combined_img.paste(vae_img, (img_width + spacing, 0))
        images.append(combined_img)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Save as GIF
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=len(images) / 60,
        loop=0,
    )
    print(f"Vision reconstruction GIF saved to {save_path}")


def create_memory_gif(
    episode: Episode,
    vision: ConvVAE,
    memory: MDN_RNN,
    device,
    save_path=Path("media/vision_memory_reconstruction.gif"),
):
    observations = episode.observations.unsqueeze(0).to(device)
    actions = episode.actions.unsqueeze(0).to(device)

    # Get latent representations from VAE
    latents = vision.get_batched_latents(observations)

    # Initialize RNN hidden state
    hidden_state, cell_state = memory.init_hidden()
    hidden_state = hidden_state.to(device)
    cell_state = cell_state.to(device)

    # Generate predictions using MDN-RNN
    predicted_latents = []
    for t in range(latents.shape[1] - 1):
        pi, mu, sigma, hidden_state, cell_state = memory(
            latents[:, t, :], actions[:, t, :], None, None
        )
        predicted_latent = memory.sample_latent(pi, mu, sigma)
        predicted_latents.append(predicted_latent)

    predicted_latents = torch.stack(predicted_latents, dim=1)

    # Decode the latents
    vae_reconstructions = vision.decoder(latents.squeeze(0))
    memory_reconstructions = vision.decoder(predicted_latents.squeeze(0))

    # Set up visualization parameters
    scale_factor = 1
    spacing = 1
    img_width, img_height = 64 * scale_factor, 64 * scale_factor
    total_width = img_width * 3 + spacing * 3
    total_height = img_height

    images = []

    for t in range(vae_reconstructions.shape[0] - 1):
        original_img = T.Resize((img_height, img_width))(
            T.ToPILImage()(observations[0, t].cpu())
        )
        vision_img = T.Resize((img_height, img_width))(
            T.ToPILImage()(vae_reconstructions[t].cpu())
        )
        memory_img = T.Resize((img_height, img_width))(
            T.ToPILImage()(memory_reconstructions[t].cpu())
        )

        combined_img = Image.new("RGB", (total_width, total_height), (0, 0, 0))
        combined_img.paste(original_img, (0, 0))
        combined_img.paste(vision_img, (img_width + spacing, 0))
        combined_img.paste(memory_img, (2 * (img_width + spacing), 0))
        images.append(combined_img)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as GIF
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=len(images) / 60,
        loop=0,
    )
    print(f"Vision and Memory reconstruction GIF saved to {save_path}")


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
        vision_batch_size=64,
        vision_epochs=0,
        memory_batch_size=32,
        memory_epochs=20,
        controller_epochs=50,
        controller_max_steps=500,
        population_size=16,
    ):
        rollout_dataset = RolloutDataset(
            num_rollouts, max_steps, continuous=self.continuous
        )

        (
            train_episodes,
            test_episodes,
            val_episodes,
        ) = torch.utils.data.random_split(rollout_dataset, [0.7, 0.2, 0.1])
        train_dataset = RolloutDataset.from_subset(train_episodes)
        test_dataset = RolloutDataset.from_subset(test_episodes)
        val_dataset = RolloutDataset.from_subset(val_episodes)
        train_dataloader = RolloutDataloader(
            train_dataset, vision_batch_size, shuffle=True
        )
        test_dataloader = RolloutDataloader(
            test_dataset, vision_batch_size, shuffle=True
        )
        val_dataloader = RolloutDataloader(val_dataset, vision_batch_size, shuffle=True)

        vision_trainer = VisionTrainer(self.vision)
        vision_trainer.train(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            val_dataloader=val_dataloader,
            epochs=vision_epochs,
            optimizer=torch.optim.Adam(self.vision.parameters()),
        )
        idxs = [np.random.randint(0, num_rollouts) for _ in range(5)]
        # Only create the gifs to check actual visual improvements
        if vision_epochs > 0:
            for idx in idxs:
                episode = Episode.load(rollout_dataset[idx])
                create_vision_gif(
                    episode,
                    self.vision,
                    self.device,
                    Path(f"media/vision/vision_reconstruction_{idx}.gif"),
                )

        latent_training_set = LatentDataset(train_dataset, self.vision)
        latent_test_set = LatentDataset(test_dataset, self.vision)
        latent_val_set = LatentDataset(val_dataset, self.vision)

        train_dataloader = LatentDataloader(
            latent_training_set, memory_batch_size, shuffle=True
        )
        test_dataloader = LatentDataloader(
            latent_test_set, memory_batch_size, shuffle=True
        )
        val_dataloader = LatentDataloader(
            latent_val_set, memory_batch_size, shuffle=True
        )

        memory_trainer = MemoryTrainer(self.memory)
        memory_trainer.train(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            val_dataloader=val_dataloader,
            optimizer=torch.optim.Adam(self.memory.parameters()),
            epochs=memory_epochs,
        )
        if memory_epochs > 0:
            for idx in idxs:
                episode = Episode.load(rollout_dataset[idx])
                create_memory_gif(
                    episode,
                    self.vision,
                    self.memory,
                    self.device,
                    Path(f"media/memory/memory_vision_reconstruction_{idx}.gif"),
                )

        controller_trainer = ControllerTrainer(
            self.controller, self.vision, self.memory, population_size=population_size
        )
        controller_trainer.train(controller_epochs, controller_max_steps)

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
    # policy.train(
    #     num_rollouts=100,
    #     max_steps=30,
    #     batch_size=32,
    #     vision_epochs=5,
    #     memory_epochs=5,
    #     controller_epochs=10,
    #     population_size=16,
    # )
    policy.train()
