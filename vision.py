import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# Test imports
import matplotlib.pyplot as plt

from dataset import Rollout, RolloutDataloader, RolloutDataset
from pathlib import Path


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
        latents = []
        mu, log_sigma = self.encoder(observation)
        sigma = log_sigma.exp()
        return mu + sigma * torch.randn_like(sigma)

    def get_latents(self, observations: torch.Tensor) -> torch.Tensor:
        latents = []
        print(f"{observations.shape=}")
        for observation in observations:
            mu, log_sigma = self.encoder(observation)
            sigma = log_sigma.exp()
            latents.append(mu + sigma * torch.randn_like(sigma))
        return torch.stack(latents)

    # This was taken directly from the ofifcial pytorch example repository:
    # https://github.com/pytorch/examples/blob/1bef748fab064e2fc3beddcbda60fd51cb9612d2/vae/main.py#L81
    def loss(self, reconstruction, original, mu, log_sigma) -> torch.Tensor:
        bce = F.mse_loss(input=reconstruction, target=original, reduction="sum")
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.sum(1 + 2 * log_sigma - mu.pow(2) - (2 * log_sigma).exp())
        return bce + kld

    def train_and_test(self, dataloader: RolloutDataloader, epochs: int = 10):
        super().train()
        optimizer = torch.optim.Adam(self.parameters())

        for epoch in range(epochs):
            train_loss = 0
            for batch_rollouts_observations, _, _ in dataloader:
                # Move to the desired device, make device agnostic
                batch_rollouts_observations = batch_rollouts_observations.to(
                    next(self.parameters()).device
                )
                # Keep the original information about the shape
                original_shape = batch_rollouts_observations.shape
                # Make a long tensor containing all the observations
                full_batch_observations = batch_rollouts_observations.reshape(
                    batch_rollouts_observations.shape[0]
                    * batch_rollouts_observations.shape[1],
                    *batch_rollouts_observations.shape[2:],
                )

                # "break" the information relative to the sequentiality and shuffle

                shuffled_indices = torch.randperm(full_batch_observations.shape[0])
                shuffled_full_bro = full_batch_observations[shuffled_indices]
                # Make the resulting shuffled observations a tensor of shape
                # number of elements in an episode x batch_size x observation shape
                shuffled_full_bro = shuffled_full_bro.reshape(
                    original_shape[1], original_shape[0], *original_shape[2:]
                )
                # This will make ~max_episode iterations, depending on effectively the lenght of
                # each episode
                for batch_observations in shuffled_full_bro:
                    optimizer.zero_grad()
                    reconstruction, mu, log_sigma = self(batch_observations)
                    loss = self.__loss(
                        reconstruction, batch_observations, mu, log_sigma
                    )
                    loss.backward()
                    # This loss is now relative to batch_size elements
                    train_loss += loss.item() / shuffled_full_bro.shape[1]
                    optimizer.step()
                # Finally we normalize again because the previous loop was done
                # on a number of elements equal to number of elements in an episode
                train_loss /= shuffled_full_bro.shape[0]
            print(f"Epoch {epoch+1} | loss {train_loss}")
        torch.save(self.state_dict(), Path("models") / "vision.pt")

    @staticmethod
    def from_pretrained(file_path: Path):
        loaded_data = torch.load(file_path, weights_only=False)
        conv_vae = ConvVAE()
        conv_vae.load_state_dict(loaded_data)
        return conv_vae

    def _check(self, image: torch.Tensor):
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


class VisionTrainer:
    def __init__(self, dataset: RolloutDataset) -> None:
        self.training_set, self.test_set, self.validation_set = (
            torch.utils.data.random_split(dataset, [0.5, 0.3, 0.2])
        )
    def _train_step(self,vision,dataloader, optimizer):
        vision.train()
        train_loss = 
    def train(self, vision: ConvVAE, epochs:int, optimizer, savepath):


if __name__ == "__main__":
    file_path = Path("data") / "dataset.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if file_path.exists():
        dataset = RolloutDataset.load(file_path=file_path)
    else:
        dataset = RolloutDataset(num_rollouts=10, max_steps=10)
        dataset.save(file_path=file_path)

    dataloader = RolloutDataloader(dataset, 2)

    conv_vae = ConvVAE().to(device)
    conv_vae.train_and_test(
        dataloader=dataloader,
        epochs=20,
    )
