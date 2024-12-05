from pathlib import Path
from typing import Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim.adam
from latent_dataset import LatentDataloader
import torchvision.transforms as T
from PIL import Image


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MDN_RNN(nn.Module):
    def __init__(
        self,
        latent_dimension: int = 32,
        hidden_units: int = 256,
        num_mixtures: int = 5,
        continuos=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_units
        self.num_mixtures = num_mixtures
        self.latent_dimension = latent_dimension
        self.continuos = continuos
        self.rnn = nn.LSTM(
            latent_dimension + (3 if continuos else 1), hidden_units, batch_first=True
        )
        self.fc_pi = nn.Linear(hidden_units, num_mixtures)
        self.fc_mu = nn.Linear(hidden_units, num_mixtures * latent_dimension)
        self.fc_log_sigma = nn.Linear(hidden_units, num_mixtures * latent_dimension)

    def forward(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]
    ]:
        if not self.continuos:
            actions = actions.unsqueeze(-1)
        rnn_out, hidden = self.rnn(
            torch.cat([latents, actions], dim=-1),
            hidden,
        )
        pi = F.softmax(self.fc_pi(rnn_out), dim=-1)
        mu = self.fc_mu(rnn_out)
        log_sigma = self.fc_log_sigma(rnn_out)
        sigma = torch.exp(log_sigma)
        return pi, mu, sigma, hidden  # type:ignore

    def __call__(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]
    ]:
        return self.forward(latents, actions, hidden)

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
        model_path: Path = Path("models/memory_discrete.pt"),
    ) -> "MDN_RNN":
        if not model_path.exists():
            raise FileNotFoundError(f"Couldn't find the Mdn-RNN model at {model_path}")
        loaded_data = torch.load(model_path, weights_only=True)
        mdn_rnn = MDN_RNN(continuos="continuos" in model_path.name)
        mdn_rnn.load_state_dict(loaded_data)
        return mdn_rnn


class MemoryTrainer:
    def _train_step(
        self,
        memory: MDN_RNN,
        train_dataloader: LatentDataloader,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        memory.train()
        train_loss = 0
        for (
            batch_latent_episodes_observations,
            batch_latent_episodes_actions,
            _,
        ) in train_dataloader:
            device = next(memory.parameters()).device
            batch_latent_episodes_observations = batch_latent_episodes_observations.to(
                device
            )
            batch_latent_episodes_actions = batch_latent_episodes_actions.to(device)
            target = batch_latent_episodes_observations[:, 1:, :]
            pi, mu, sigma, _ = memory(
                batch_latent_episodes_observations[:, :-1],
                batch_latent_episodes_actions[:, :-1],
            )
            loss = memory.loss(pi, mu, sigma, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        return train_loss

    def _test_step(
        self,
        memory: MDN_RNN,
        test_dataloader: LatentDataloader,
    ) -> float:
        memory.eval()
        test_loss = 0

        for (
            batch_latent_episodes_observations,
            batch_latent_episodes_actions,
            _,
        ) in test_dataloader:
            # Move data to the correct device
            device = next(memory.parameters()).device
            batch_latent_episodes_observations = batch_latent_episodes_observations.to(
                device
            )
            batch_latent_episodes_actions = batch_latent_episodes_actions.to(device)
            target = batch_latent_episodes_observations[:, 1:, :]
            pi, mu, sigma, _ = memory(
                batch_latent_episodes_observations[:, :-1],
                batch_latent_episodes_actions[:, :-1],
            )
            loss = memory.loss(pi, mu, sigma, target)
            test_loss += loss.item()
        test_loss /= len(test_dataloader)
        return test_loss

    def train(
        self,
        memory: MDN_RNN,
        train_dataloader: LatentDataloader,
        test_dataloader: LatentDataloader,
        optimizer: torch.optim.Optimizer,
        val_dataloader: Optional[LatentDataloader] = None,
        epochs: int = 10,
        save_path: Path = Path("models/memory.pt"),
    ):
        # if memory.continuos and not train_dataloader.dataset.continuos:  # type: ignore
        #     raise ValueError(
        #         f"Cannot train a {'continuos' if memory.continuos else 'discrete'} {type(memory)}"
        #         + f"using a {'continuos' if train_dataloader.dataset.continuos else 'discrete'} {type(train_dataloader.dataset)}"  # type: ignore
        #     )
        # if memory.continuos and not test_dataloader.dataset.continuos:  # type: ignore
        #     raise ValueError(
        #         f"Cannot train a {'continuos' if memory.continuos else 'discrete'} {type(memory)}"
        #         + f"using a {'continuos' if test_dataloader.dataset.continuos else 'discrete'} {type(test_dataloader.dataset)}"  # type: ignore
        #     )
        # if memory.continuos and not val_dataloader.dataset.continuos:  # type: ignore
        #     raise ValueError(
        #         f"Cannot train a {'continuos' if memory.continuos else 'discrete'} {type(memory)}"
        #         + f"using a {'continuos' if val_dataloader.dataset.continuos else 'discrete'} {type(val_dataloader.dataset)}"  # type: ignore
        #     )
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            train_loss = self._train_step(memory, train_dataloader, optimizer)
            test_loss = self._test_step(memory, test_dataloader)
            # scheduler.step()

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f}"
            )
        if val_dataloader is not None:
            val_loss = self._test_step(memory, test_dataloader)
            print(f"Validation Loss: {val_loss:.4f}")
        # Save the model
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(memory.state_dict(), save_path)
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    from dataset import RolloutDataset
    from latent_dataset import LatentDataset, LatentDataloader
    from vision import ConvVAE

    CONTINUOS = True
    rollout_dataset = RolloutDataset(
        "create",
        num_rollouts=10,
        max_steps=10,
        continuos=CONTINUOS,
    )

    vision = ConvVAE.from_pretrained().to(DEVICE)

    train_episodes, test_episodes, val_episodes = torch.utils.data.random_split(
        rollout_dataset, [0.5, 0.3, 0.2]
    )
    training_set = LatentDataset(
        RolloutDataset(
            "from",
            episodes=[
                rollout_dataset.episodes_paths[idx] for idx in train_episodes.indices
            ],
            continuos=CONTINUOS,
        ),
        vision,
        "create",
    )
    test_set = LatentDataset(
        RolloutDataset(
            "from",
            episodes=[
                rollout_dataset.episodes_paths[idx] for idx in test_episodes.indices
            ],
            continuos=CONTINUOS,
        ),
        vision,
        "create",
    )
    val_set = LatentDataset(
        RolloutDataset(
            "from",
            episodes=[
                rollout_dataset.episodes_paths[idx] for idx in val_episodes.indices
            ],
            continuos=CONTINUOS,
        ),
        vision,
        "create",
    )

    train_dataloader = LatentDataloader(training_set, 64)
    test_dataloader = LatentDataloader(test_set, 64)
    test_dataloader = LatentDataloader(val_set, 64)
    memory = MDN_RNN(continuos=CONTINUOS)
    memory_trainer = MemoryTrainer()
    memory_trainer.train(
        memory,
        train_dataloader,
        test_dataloader,
        torch.optim.Adam(memory.parameters()),
        save_path=Path("models/memory_continuos.pt"),
    )
