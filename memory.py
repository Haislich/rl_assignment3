import torch
import torch.nn as nn
import torch.nn.functional as F


class MDN_RNN(nn.Module):
    def __init__(self, latent_dimension=32, hidden_units=256, num_mixtures=5):
        super().__init__()
        self.hidden_dim = hidden_units
        self.num_mixtures = num_mixtures

        self.rnn = nn.LSTM(latent_dimension, hidden_units, batch_first=True)
        # The MDN is trying to predict the probability distribution of the next latent vector.
        # Instead of predicting a single point value (like a regression model),
        # the MDN models the output as a Mixture of Gaussian Distributions (MoG).
        # This allows the model to handle multimodal data, where multiple outcomes are possible.
        # K for $\pi$ mixing coefficients
        # K * latetent_dimension for the means
        # K * latetent_dimension for the standard deviations

        self.mdn = nn.Linear(hidden_units, num_mixtures * (2 * latent_dimension + 1))

    def forward(self, latent, hidden=None, temperature=None) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        rnn_out, hidden = self.rnn(latent, hidden)  # LSTM output

        # MDN layer: outputs mixture weights (pi), means (mu), variances (sigma)
        mdn_params = self.mdn(rnn_out)
        # Remove sequence dimension
        mdn_params = mdn_params.view(
            -1, self.num_mixtures, 2 * self.latent_dimension + 1
        )

        # As described in: https://arxiv.org/pdf/1308.0850 (19,20)
        # The mixture weight outputs are normalised with a softmax
        # function to ensure they form a valid discrete distribution,
        # and the other outputs are passed through suitable functions to keep
        # their values within meaningful range (for example the exponential function is
        # typically applied to outputs used as scale parameters, which must be positive)
        # As described in: https://arxiv.org/pdf/1704.03477 (4)
        # We can control the level of randomness we would like our samples to have during the
        # sampling process by introducing a temperature parameter $\tau$.
        # We can scale the softmax parameters of the categorial distribution
        # and also the $\sigma$ parameters of the bivariate normal distribution
        # by a temperature parameter $\tau$ , to control the level of randomness in our samples.
        pi = F.softmax(mdn_params[:, :, 0] / temperature, dim=-1)
        # The means are taken as is
        mu: torch.Tensor = mdn_params[:, :, 1 : self.latent_dimension + 1]
        # The sigmas are exponentiated, because this guarantees positivity
        sigma = torch.exp(mdn_params[:, :, self.latent_dimension + 1 :])
        return pi, mu, sigma, hidden

    def train(self,dataloader, epochs:int =10):
        for 