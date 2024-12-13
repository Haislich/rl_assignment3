from pathlib import Path
from typing import Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim.adam


class MDN_RNN(nn.Module):
    def __init__(
        self,
        latent_dimension: int = 32,
        hidden_units: int = 256,
        num_mixtures: int = 5,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        self.to(device)
        self.device = device
        self.hidden_dim = hidden_units
        self.num_mixtures = num_mixtures
        self.latent_dimension = latent_dimension
        self.rnn = nn.LSTM(latent_dimension + 3, hidden_units, batch_first=True)
        self.fc_pi = nn.Linear(hidden_units, num_mixtures)
        self.fc_mu = nn.Linear(hidden_units, num_mixtures * latent_dimension)
        self.fc_log_sigma = nn.Linear(hidden_units, num_mixtures * latent_dimension)

    def init_hidden(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_state = torch.zeros(1, 1, self.hidden_dim).to(self.device)
        cell_state = torch.zeros(1, 1, self.hidden_dim).to(self.device)
        return hidden_state, cell_state

    def forward(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        cell_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

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
        sigma = torch.clamp(sigma, min=1e-8)

        normal = torch.distributions.Normal(loc=mu, scale=sigma)
        log_probs = normal.log_prob(target).sum(dim=-1)
        log_pi = torch.log(pi + 1e-8)
        log_probs = -torch.logsumexp(log_pi + log_probs, dim=-1)
        return log_probs.mean()

    def sample_latent(
        self, pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:

        batch_size = mu.size(0)
        mu = mu.view(batch_size, self.latent_dimension, self.num_mixtures)
        sigma = sigma.view(batch_size, self.latent_dimension, self.num_mixtures)
        categorical = torch.distributions.Categorical(pi)
        mixture_indices = categorical.sample()
        mixture_indices = mixture_indices.unsqueeze(1).expand(-1, self.latent_dimension)

        selected_mu = torch.gather(mu, 2, mixture_indices.unsqueeze(-1)).squeeze(-1)
        selected_sigma = torch.gather(sigma, 2, mixture_indices.unsqueeze(-1)).squeeze(
            -1
        )
        normal = torch.distributions.Normal(selected_mu, selected_sigma)
        return normal.rsample()

    @staticmethod
    def from_pretrained(
        device,
        model_path: Path = Path("models/memory.pt"),
    ) -> "MDN_RNN":
        mdn_rnn = MDN_RNN(device=device)
        if model_path.exists():
            loaded_data = torch.load(model_path, weights_only=True, map_location=device)
            mdn_rnn.load_state_dict(loaded_data["model_state"])
        return mdn_rnn
