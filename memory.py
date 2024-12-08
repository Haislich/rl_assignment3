from pathlib import Path
from typing import Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim.adam
from torch.utils.tensorboard.writer import SummaryWriter

from tqdm import tqdm
from latent_dataset import LatentDataloader
from math import ceil


class MDN_RNN(nn.Module):
    def __init__(
        self,
        latent_dimension: int = 32,
        hidden_units: int = 256,
        num_mixtures: int = 5,
        continuous=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_units
        self.num_mixtures = num_mixtures
        self.latent_dimension = latent_dimension
        self.continuous = continuous
        self.rnn = nn.LSTM(
            latent_dimension + (3 if continuous else 1), hidden_units, batch_first=True
        )
        self.fc_pi = nn.Linear(hidden_units, num_mixtures)
        self.fc_mu = nn.Linear(hidden_units, num_mixtures * latent_dimension)
        self.fc_log_sigma = nn.Linear(hidden_units, num_mixtures * latent_dimension)

    def init_hidden(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_state = torch.zeros(1, 1, self.hidden_dim)
        cell_state = torch.zeros(1, 1, self.hidden_dim)
        return hidden_state, cell_state

    def forward(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        cell_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.continuous:
            actions = actions.unsqueeze(-1)
        rnn_out, (hidden_state, _cell_state) = self.rnn(
            torch.cat([latents, actions], dim=-1),
            (
                None
                if hidden_state is None or cell_state is None
                else (hidden_state, cell_state)
            ),
        )

        pi = F.softmax(self.fc_pi(rnn_out), dim=-1)
        mu = self.fc_mu(rnn_out)
        log_sigma = self.fc_log_sigma(rnn_out)
        sigma = torch.exp(log_sigma)
        return pi, mu, sigma, hidden_state, cell_state  # type:ignore

    def loss(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, seq_len = target.shape[:2]
        mu = mu.view(batch_size, seq_len, self.num_mixtures, self.latent_dimension)
        sigma = sigma.view(
            batch_size, seq_len, self.num_mixtures, self.latent_dimension
        )
        target = target.unsqueeze(2).expand(-1, -1, self.num_mixtures, -1)
        sigma = torch.clamp(sigma, min=1e-4)

        normal = torch.distributions.Normal(loc=mu, scale=sigma)
        log_probs = normal.log_prob(target).sum(dim=-1)
        log_pi = torch.log(pi + 1e-4)
        log_probs = -torch.logsumexp(log_pi + log_probs, dim=-1)
        return log_probs.mean()

    def sample_latent(
        self, pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:

        batch_size = mu.size(0)
        mu = mu.view(batch_size, self.latent_dimension, self.num_mixtures)
        sigma = sigma.view(batch_size, self.latent_dimension, self.num_mixtures)
        categorical = torch.distributions.Categorical(pi)
        mixture_indices = categorical.sample()  # Shape: (batch_size,)
        mixture_indices = mixture_indices.unsqueeze(1).expand(-1, self.latent_dimension)

        selected_mu = torch.gather(mu, 2, mixture_indices.unsqueeze(-1)).squeeze(-1)
        selected_sigma = torch.gather(sigma, 2, mixture_indices.unsqueeze(-1)).squeeze(
            -1
        )
        normal = torch.distributions.Normal(selected_mu, selected_sigma)
        return normal.rsample()

    @staticmethod
    def from_pretrained(
        model_path: Path = Path("models/memory_continuous.pt"),
    ) -> "MDN_RNN":
        if not model_path.exists():
            raise FileNotFoundError(f"Couldn't find the Mdn-RNN model at {model_path}")
        loaded_data = torch.load(model_path, weights_only=True)
        mdn_rnn = MDN_RNN(continuous="continuous" in model_path.name)
        mdn_rnn.load_state_dict(loaded_data["model_state"])
        return mdn_rnn


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
        log_dir=Path("logs/memory"),
    ):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
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
        ):
            print(f"Epoch {epoch + 1}/{epochs}")
            train_loss = self._train_step(
                train_dataloader,
                optimizer,
            )
            test_loss = self._test_step(test_dataloader)

            # Log to TensorBoard
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Test", test_loss, epoch)
            print()
            print(
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
