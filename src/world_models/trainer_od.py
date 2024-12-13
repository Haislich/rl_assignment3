from pathlib import Path
from world_models.dataset import RolloutDataloader
from world_models.models.memory import MDN_RNN
from world_models.models.vision import ConvVAE
from tqdm import tqdm
import torch
from typing import Optional


class VisionTrainer:
    def __init__(self, vision: ConvVAE) -> None:
        self.vision = vision
        self.device = next(self.vision.parameters()).device

    def _train_step(
        self,
        train_dataloader: RolloutDataloader,
        optimizer: torch.optim.Optimizer,
        beta: float = 0.1,
    ) -> float:
        self.vision.train()
        train_loss = 0

        for batch_episodes_observations, _, _ in tqdm(
            train_dataloader,
            total=len(train_dataloader),
            desc="Loading episode batches from the dataloader",
            leave=False,
        ):

            batch_size, seq_len, *obs_shape = batch_episodes_observations.shape

            # Step 1: Reshape the tensor to (batch_size * seq_len, *obs_shape)
            compacted_observations = batch_episodes_observations.view(-1, *obs_shape)

            # Step 2: Shuffle along the first dimension
            shuffled_indices = torch.randperm(compacted_observations.size(0))
            shuffled_observations = compacted_observations[shuffled_indices]

            # Step 3: Reshape back to (seq_len, batch_size, *obs_shape)
            shuffled_episodes_batched_observations = shuffled_observations.view(
                seq_len, batch_size, *obs_shape
            ).to(self.device)
            for batched_observations in tqdm(
                shuffled_episodes_batched_observations,
                total=shuffled_episodes_batched_observations.shape[0],
                desc="Processing timesteps for current episode batch",
                leave=False,
            ):
                reconstruction, mu, log_sigma = self.vision.forward(
                    batched_observations
                )
                loss = self.vision.loss(
                    reconstruction, batched_observations, mu, log_sigma, beta
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= batch_episodes_observations.shape[0]
        train_loss /= len(train_dataloader)
        return train_loss

    def _test_step(
        self, test_dataloader: RolloutDataloader, beta: float = 0.1
    ) -> float:
        self.vision.eval()
        test_loss = 0
        with torch.inference_mode():
            for batch_episodes_observations, _, _ in tqdm(
                test_dataloader,
                total=len(test_dataloader),
                desc="Loading episode batches from the dataloader",
                leave=False,
            ):
                batch_size, seq_len, *obs_shape = batch_episodes_observations.shape
                compacted_observations = batch_episodes_observations.view(
                    -1, *obs_shape
                )
                shuffled_indices = torch.randperm(compacted_observations.size(0))
                shuffled_observations = compacted_observations[shuffled_indices]
                shuffled_episodes_batched_observations = shuffled_observations.view(
                    seq_len, batch_size, *obs_shape
                ).to(self.device)
                for batched_observations in tqdm(
                    shuffled_episodes_batched_observations,
                    total=shuffled_episodes_batched_observations.shape[0],
                    desc="Testing timesteps for current episode batch",
                    leave=False,
                ):
                    reconstruction, mu, log_sigma = self.vision.forward(
                        batched_observations
                    )
                    loss = self.vision.loss(
                        reconstruction, batched_observations, mu, log_sigma, beta
                    )
                    test_loss += loss.item()
                test_loss /= batch_episodes_observations.shape[0]
            test_loss /= len(test_dataloader)
        return test_loss

    def train(
        self,
        train_dataloader: RolloutDataloader,
        test_dataloader: RolloutDataloader,
        optimizer: torch.optim.Optimizer,
        val_dataloader: Optional[RolloutDataloader] = None,
        epochs: int = 10,
        save_path=Path("models/vision.pt"),
    ):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        initial_epoch = 0
        if save_path.exists():
            vision_metadata = torch.load(
                save_path, weights_only=True, map_location=self.device
            )
            initial_epoch = vision_metadata["epoch"]
            self.vision.load_state_dict(vision_metadata["model_state"])
            optimizer.load_state_dict(vision_metadata["optimizer_state"])
        for epoch in tqdm(
            range(initial_epoch, epochs + initial_epoch),
            total=epochs,
            desc="Training Vision",
            leave=False,
        ):
            beta = 1.0  # min(1.0, epoch / epochs)
            train_loss = self._train_step(
                train_dataloader,
                optimizer,
                beta,
            )
            test_loss = self._test_step(test_dataloader, beta)

            tqdm.write(
                f"\tEpoch {epoch + 1}/{epochs+initial_epoch} | "
                f"Train Loss: {train_loss:.5f} | "
                f"Test Loss: {test_loss:.5f}"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": self.vision.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                save_path,
            )

        if val_dataloader is not None:
            val_loss = self._test_step(test_dataloader)
            print(f"Validation Loss: {val_loss:.4f}")

        print(f"Model saved to {save_path}")


class MemoryTrainer:
    def __init__(self, memory: MDN_RNN) -> None:
        self.memory = memory
        self.device = next(self.memory.parameters()).device

    def _train_step(
        self,
        train_dataloader: LatentDataloader,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        self.memory.train()
        train_loss = 0
        for (
            batch_latent_episodes_observations,
            batch_latent_episodes_actions,
            _,
        ) in tqdm(
            train_dataloader,
            total=ceil(
                len(train_dataloader) / train_dataloader.batch_size  # type:ignore
            ),
            desc="Processing latent episode batches",
            leave=False,
        ):
            batch_latent_episodes_observations = batch_latent_episodes_observations.to(
                self.device
            )
            batch_latent_episodes_actions = batch_latent_episodes_actions.to(
                self.device
            )
            target = batch_latent_episodes_observations[:, 1:, :]
            pi, mu, sigma, *_ = self.memory.forward(
                batch_latent_episodes_observations[:, :-1],
                batch_latent_episodes_actions[:, :-1],
            )
            loss = self.memory.loss(pi, mu, sigma, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        return train_loss

    def _test_step(
        self,
        test_dataloader: LatentDataloader,
    ) -> float:
        self.memory.eval()
        test_loss = 0
        with torch.inference_mode():
            for (
                batch_latent_episodes_observations,
                batch_latent_episodes_actions,
                _,
            ) in tqdm(
                test_dataloader,
                total=ceil(
                    len(test_dataloader) / test_dataloader.batch_size  # type:ignore
                ),
                desc="Testing on latent episode batches",
                leave=False,
            ):
                batch_latent_episodes_observations = (
                    batch_latent_episodes_observations.to(self.device)
                )
                batch_latent_episodes_actions = batch_latent_episodes_actions.to(
                    self.device
                )

                target = batch_latent_episodes_observations[:, 1:, :]
                pi, mu, sigma, *_ = self.memory.forward(
                    batch_latent_episodes_observations[:, :-1],
                    batch_latent_episodes_actions[:, :-1],
                )
                loss = self.memory.loss(pi, mu, sigma, target)
                test_loss += loss.item()
        test_loss /= len(test_dataloader)
        return test_loss

    def train(
        self,
        train_dataloader: LatentDataloader,
        test_dataloader: LatentDataloader,
        optimizer: torch.optim.Optimizer,
        val_dataloader: Optional[LatentDataloader] = None,
        epochs: int = 10,
        save_path: Path = Path("models/memory_continuous.pt"),
    ):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        initial_epoch = 0
        if save_path.exists():
            memory_metadata = torch.load(save_path, weights_only=True)
            initial_epoch = memory_metadata["epoch"]
            self.memory.load_state_dict(memory_metadata["model_state"])
            optimizer.load_state_dict(memory_metadata["optimizer_state"])
        for epoch in tqdm(
            range(initial_epoch, epochs + initial_epoch),
            total=epochs,
            desc="Training Memory",
            leave=False,
        ):
            train_loss = self._train_step(
                train_dataloader,
                optimizer,
            )
            test_loss = self._test_step(test_dataloader)

            tqdm.write(
                f"\tEpoch {epoch + 1}/{epochs+initial_epoch} | "
                f"Train Loss: {train_loss:.5f} | "
                f"Test Loss: {test_loss:.5f}"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": self.memory.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                save_path,
            )
        if val_dataloader is not None:
            val_loss = self._test_step(test_dataloader)
            print(f"Validation Loss: {val_loss:.5f}")
        print(f"Model saved to {save_path}")


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
                    # _mu, _pi, _sigma, hidden_state, cell_state = self.memory.forward(
                    #     latent_observation,
                    #     action,
                    #     hidden_state,
                    #     cell_state,
                    # )
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

        initial_solution = self.controller.get_weights()
        bestfit = float("-inf")
        if save_path.exists():
            controller_metadata = torch.load(save_path, weights_only=True)
            self.controller.load_state_dict(controller_metadata["model_state"])
            initial_solution = self.controller.get_weights()
            bestfit = controller_metadata["bestfit"]
        solver = CMAEvolutionStrategy(
            initial_solution, 0.2, {"popsize": self.population_size}
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
