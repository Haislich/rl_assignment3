import torch
import torch.nn as nn
import torch.nn.functional as F

class MDNRNN(nn.Module):
    def __init__(self, latent_dimension =32, hidden_units= 256, num_mixtures= 5):
        super().__init__()
        self.hidden_dim = hidden_units
        self.num_mixtures = num_mixtures


        self.lstm = nn.LSTM(latent_dimension, hidden_units, batch_first=True)
        # The MDN is trying to predict the probability distribution of the next latent vector.
        # Instead of predicting a single point value (like a regression model),
        # the MDN models the output as a Mixture of Gaussian Distributions (MoG).
        # This allows the model to handle multimodal data, where multiple outcomes are possible.
        # K for $\pi$ mixing coefficients
        # K * latetent_dimension for the means
        # K * latetent_dimension for the standard deviations

        self.mdn = nn.Linear(hidden_units, num_mixtures * (2 * latent_dimension + 1))

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)  # LSTM output

        # MDN layer: outputs mixture weights (pi), means (mu), variances (sigma)
        mdn_params = self.mdn(lstm_out)
        mdn_params = mdn_params.view(-1, self.num_mixtures, 2 * self.latent_dimension + 1)

        pi = F.softmax(mdn_params[:, :, 0], dim=-1)
        mu = mdn_params[:, :, 1:self.latent_dimension + 1]  # Means
        sigma = torch.exp(mdn_params[:, :, self.latent_dimension + 1:])  # Diagonal variance

        return pi, mu, sigma, hidden

