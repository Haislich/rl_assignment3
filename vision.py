import torch
from torch import device, nn
import torch.nn.functional as F
import numpy as np

# Test imports
import matplotlib.pyplot as plt

from dataset import Rollout, RolloutDataloader, RolloutDataset


class Encoder(nn.Module):
    def __init__(self, latent_dimension, *, stride=2):
        super().__init__()
        self.relu_conv1 = nn.Conv2d(3, 32, 4, stride=stride)
        self.relu_conv2 = nn.Conv2d(32, 64, 4, stride=stride)
        self.relu_conv3 = nn.Conv2d(64, 128, 4, stride=stride)
        self.relu_conv4 = nn.Conv2d(128, 256, 4, stride=stride)

        self.fc_mu = nn.Linear(2 * 2 * 256, latent_dimension)
        self.fc_sigma = nn.Linear(2 * 2 * 256, latent_dimension)

    def forward(self, x):
        x = F.relu(self.relu_conv1(x))
        x = F.relu(self.relu_conv2(x))
        x = F.relu(self.relu_conv3(x))
        x = F.relu(self.relu_conv4(x))
        # Flatten the tensor for dense layers
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_sigma = self.fc_sigma(x)

        # This reparametrization trick allow us to still be able to train it
        return mu, log_sigma


class Decoder(nn.Module):
    def __init__(self, latent_dimension, image_chanels, *, stride=2):
        super().__init__()
        self.fc = nn.Linear(latent_dimension, 1024)
        self.relu_deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=stride)
        self.relu_deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=stride)
        self.relu_deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=stride)
        self.sigmoid_deconv = nn.ConvTranspose2d(32, image_chanels, 6, stride=stride)

    def forward(self, x):
        x = self.fc(x)
        x = x.unsqueeze(-1).unsqueeze(-1)

        x = F.relu(self.relu_deconv1(x))
        x = F.relu(self.relu_deconv2(x))
        x = F.relu(self.relu_deconv3(x))
        # https://github.com/pytorch/pytorch/issues/65910
        return torch.sigmoid(self.sigmoid_deconv(x))


class ConvVAE(nn.Module):
    def __init__(self, latent_dimension=32, image_channels=3):
        super().__init__()
        self.latent_dimension = latent_dimension
        # https://worldmodels.github.io/#:~:text=each%20convolution%20and%20deconvolution%20layer%20uses%20a%20stride%20of%202.
        self.encoder = Encoder(latent_dimension, stride=2)
        self.decoder = Decoder(latent_dimension, image_channels, stride=2)

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_sigma = self.encoder(x)
        sigma = log_sigma.exp()
        z = mu + sigma * torch.randn_like(sigma)
        reconstruction = self.decoder(z)
        return reconstruction, mu, sigma

    def get_latent(self, observation: torch.Tensor) -> torch.Tensor:
        mu, log_sigma = self.encoder(observation)
        sigma = log_sigma.exp()
        return mu + sigma * torch.randn_like(sigma)

    # This was taken directly from the ofifcial pytorch example repository:
    # https://github.com/pytorch/examples/blob/1bef748fab064e2fc3beddcbda60fd51cb9612d2/vae/main.py#L81
    def __loss(self, reconstruction, original, mu, log_sigma) -> torch.Tensor:
        BCE = F.binary_cross_entropy(
            input=reconstruction, target=original, reduction="sum"
        )

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())

        return BCE + KLD

    def train(self, dataloader: RolloutDataloader, epochs: int = 10):
        super().train()
        train_loss = 0
        optimizer = torch.optim.Adam(self.parameters())

        for epoch in range(epochs):
            for batch_rollouts_observations, _, _ in dataloader:
                # Now we need to make the observations in a batch a single long tensor
                original_shape = batch_rollouts_observations.shape
                full_batch_observations = batch_rollouts_observations.reshape(
                    batch_rollouts_observations.shape[0]
                    * batch_rollouts_observations.shape[1],
                    *batch_rollouts_observations.shape[2:],
                )
                # "break" the information relative to the sequentiality and shuffle
                shuffled_indices = torch.randperm(full_batch_observations.shape[0])
                shuffled_full_bro = full_batch_observations[shuffled_indices]
                # for batch_observation in range(0,full_batch_observations.shape[0],dataloader.dataset.max_steps):
                shuffled_full_bro = shuffled_full_bro.reshape(*original_shape)
                for batch_observations in shuffled_full_bro:
                    optimizer.zero_grad()
                    reconstruction, mu, log_sigma = self(batch_observations)
                    loss = self.__loss(
                        reconstruction, batch_observations, mu, log_sigma
                    )
                    loss.backward()
                    train_loss += loss.item()
                    optimizer.step()
            print(f"Epoch {epoch+1} | loss {train_loss}")

    def __check(self, image: torch.Tensor):
        image = image.to(next(self.parameters()).device)

        original = image
        reconstruction, *_ = self(image)

        # Move tensors to CPU and detach for visualization
        original = original.cpu().detach().numpy()
        reconstruction = reconstruction.cpu().detach().numpy()

        # Assume the image is in the format [batch_size, channels, height, width]
        # Take the first image in the batch for visualization
        original_img = original[0]
        reconstruction_img = reconstruction[0]

        # Handle grayscale or RGB images
        if original_img.shape[0] == 1:  # Grayscale
            original_img = original_img[0]
            reconstruction_img = reconstruction_img[0]
            cmap = "gray"
        else:  # RGB
            original_img = original_img.transpose(1, 2, 0)
            reconstruction_img = reconstruction_img.transpose(1, 2, 0)
            cmap = None

        # Create the plot
        plt.figure(figsize=(8, 4))

        # Original image
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(original_img, cmap=cmap)
        plt.axis("off")

        # Reconstructed image
        plt.subplot(1, 2, 2)
        plt.title("Reconstruction")
        plt.imshow(reconstruction_img, cmap=cmap)
        plt.axis("off")

        # Show the plot
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    dataset = RolloutDataset(num_rollouts=1000, max_steps=3)
    dataloader = RolloutDataloader(dataset, 2)
    conv_vae = ConvVAE()
    conv_vae.train(
        dataloader=dataloader,
        epochs=2,
    )
    # for batch_observations, _, _ in dataloader:
    #     # Now we need to make the observations in a batch a single long tensor
    #     observations = batch_observations.reshape(
    #         batch_observations.shape[0] * batch_observations.shape[1],
    #         *batch_observations.shape[2:],
    #     )
    #     # "break" the information relative to the sequentiality and shuffle
    #     shuffled_indices = torch.randperm(observations.shape[0])
    #     shuffled_observations = observations[shuffled_indices]
