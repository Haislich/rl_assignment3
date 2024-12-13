from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from concurrent.futures import ProcessPoolExecutor
from cma import CMAEvolutionStrategy

from world_models.models.controller import Controller
from world_models.models.memory import MDN_RNN
from world_models.models.vision import ConvVAE
from world_models.policy import Policy
from tqdm import tqdm

from world_models.episode import Episode, LatentEpisode
from world_models.dataset import RolloutDataloader, RolloutDataset


@dataclass
class VisionArgs:
    epochs: int = 10
    batch_size: int = 64
    save_path = Path("./models/vision.pt")
    log_dir = Path("./logs/vision")


@dataclass
class MemoryArgs:
    epochs: int = 10
    batch_size: int = 64
    save_path = Path("./models/memory.pt")
    log_dir = Path("./logs/memory")


@dataclass
class ControllerArgs:
    num_agents: int = 12
    generations: int = 10
    episodes_per_generation: int = 3
    save_path = Path("./models/controller.pt")


class Trainer:
    def __init__(
        self,
        epochs=10,
        vision_args: VisionArgs = VisionArgs(),
        memory_args: MemoryArgs = MemoryArgs(),
        controller_args: ControllerArgs = ControllerArgs(),
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        self.epochs = epochs
        self.vision_args = vision_args
        self.memory_args = memory_args
        self.controller_args = controller_args
        self.device = device
        self.vision = ConvVAE(device=device)
        self.memory = MDN_RNN(device=device)
        self.controller = Controller(device=device)

    def rollout(self, policy):
        environment = gym.make("CarRacing-v2", render_mode="rgb_array")
        total_reward = 0.0
        episodes = []
        for _ in range(self.controller_args.episodes_per_generation):
            observation, _ = environment.reset()
            episode_reward, observations, actions, rewards = 0.0, [], [], []

            while True:
                action = policy.act(observation)
                next_observation, reward, terminated, truncated, _ = environment.step(
                    action
                )
                episode_reward += float(reward)

                if terminated or truncated:
                    break

                observations.append(policy.transformation(observation))
                actions.append(torch.from_numpy(action))
                rewards.append(torch.tensor(reward))

                observation = next_observation

            total_reward += episode_reward
            episodes.append(
                Episode(
                    torch.stack(observations),
                    torch.stack(actions),
                    torch.stack(rewards),
                )
            )

        average_reward = total_reward / self.controller_args.episodes_per_generation
        return episodes, average_reward

    def _evolve_controller(self):
        ...
    def _train_vision(
        self,
        rollout_dataloader: RolloutDataloader,
        optimizer: torch.optim.Adam,
        writer: SummaryWriter,
        trainer_epoch: int,
    ):
        self.vision.train()

        train_loss = 0.0
        initial_epoch = trainer_epoch * self.vision_args.epochs
        for epoch in tqdm(
            range(initial_epoch, initial_epoch + self.vision_args.epochs),
            desc="Training vision",
        ):
            batch_number = 0
            for batch_episodes_observations, _, _ in tqdm(
                rollout_dataloader,
                total=len(rollout_dataloader),
                leave=False,
            ):
                batch_number += 1
                batch_size, seq_len, *obs_shape = batch_episodes_observations.shape

                # Step 1: Reshape the tensor to (batch_size * seq_len, *obs_shape)
                batch_episodes_observations = batch_episodes_observations.view(
                    -1, *obs_shape
                )

                # Step 2: Shuffle along the first dimension
                shuffled_indices = torch.randperm(batch_episodes_observations.size(0))
                batch_episodes_observations = batch_episodes_observations[
                    shuffled_indices
                ]

                # Step 3: Reshape back to (seq_len, batch_size, *obs_shape)
                batch_episodes_observations = batch_episodes_observations.view(
                    seq_len, batch_size, *obs_shape
                ).to(self.device)
                for batched_observations in tqdm(
                    batch_episodes_observations,
                    total=batch_episodes_observations.shape[0],
                    leave=False,
                ):
                    reconstruction, mu, log_sigma = self.vision.forward(
                        batched_observations
                    )
                    loss = self.vision.loss(
                        reconstruction,
                        batched_observations,
                        mu,
                        log_sigma,
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                train_loss /= batch_episodes_observations.shape[0]
                tqdm.write(
                    f"Vision summary: Epoch {epoch+1} | "
                    + f"Batch number {batch_number} | "
                    + f"Train_loss {train_loss}"
                )
                writer.add_scalar(
                    "Loss/Train",
                    train_loss,
                    epoch * len(rollout_dataloader) + batch_number,
                )
        torch.save(
            {"model_state": self.vision.state_dict()}, self.vision_args.save_path
        )

    def train(self):
        self.vision_args.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.vision_args.log_dir.mkdir(parents=True, exist_ok=True)
        vision_optimizer = torch.optim.Adam(self.vision.parameters())
        vision_writer = SummaryWriter(self.vision_args.log_dir)

        self.memory_args.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_args.log_dir.mkdir(parents=True, exist_ok=True)
        memory_optimizer = torch.optim.Adam(self.memory.parameters())
        memory_writer = SummaryWriter(self.memory_args.log_dir)

        self.controller_args.save_path.parent.mkdir(parents=True, exist_ok=True)
        initial_solution = self.controller.get_weights()
        solver = CMAEvolutionStrategy(
            initial_solution, 0.2, {"popsize": self.controller_args.num_agents}
        )
        bestfit = float("-inf")
        for epoch in range(2):
            solutions = solver.ask()
            # Evolve Controller
            with ProcessPoolExecutor() as executor:
                policies = [
                    Policy(
                        self.vision,
                        self.memory,
                        Controller().set_weights(solution),
                        device=self.device,
                    )
                    for solution in solutions
                ]
                results = list(executor.map(self.rollout, policies))
            fitlist = [-reward for _, reward in results]
            solver.tell(solutions, fitlist)
            epoch_bestsol, epoch_bestfit, *_ = solver.result
            if epoch_bestfit > bestfit:
                torch.save(
                    {"model_state": self.controller.set_weights(epoch_bestsol)},
                    self.controller_args.save_path,
                )
            # Create a dataset with the new rollouts
            episodes = [
                episode for episodes_list, _ in results for episode in episodes_list
            ]
            rollout_dataset = RolloutDataset(episodes)
            rollout_dataloader = RolloutDataloader(
                rollout_dataset, batch_size=self.vision_args.batch_size
            )
            # Train vision
            self._train_vision(
                rollout_dataloader, vision_optimizer, vision_writer, epoch
            )
            # Create a latent dataset 
            latent_episodes =[
                LatentEpisode(self.vision.get_latent(episode.observation),episode.actions,episode.rewards) for episode in episodes
            ]
            latent_dataset = 
        vision_writer.close()
        memory_writer.close()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    trainer = Trainer(
        controller_args=ControllerArgs(num_agents=1),
        vision_args=VisionArgs(batch_size=2, epochs=3),
    )
    trainer.train()
