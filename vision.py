import os
from pathlib import Path

# TODO: Remove
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn

from dataset import RolloutDataloader  # , RolloutDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

    def _train_step(
        self,
        vision: ConvVAE,
        train_dataloader: RolloutDataloader,
        optimizer: torch.optim.Optimizer,
    ):
        vision.train()

        train_loss = 0
        for batch_rollouts_observations, _, _ in train_dataloader:
            # Move to the desired device, make device agnostic
            batch_rollouts_observations = batch_rollouts_observations.to(
                next(vision.parameters()).device
            )

            # Make a long tensor containing all the observations
            batch_rollouts_observations = batch_rollouts_observations.permute(
                1, 0, 2, 3, 4
            )

            # "break" the information relative to the sequentiality and shuffle

            shuffled_indices = torch.randperm(batch_rollouts_observations.shape[0])
            batch_rollouts_observations = batch_rollouts_observations[shuffled_indices]
            # Make the resulting shuffled observations a tensor of shape
            # number of elements in an episode x batch_size x observation shape
            batch_rollouts_observations = batch_rollouts_observations.permute(
                1, 0, 2, 3, 4
            )
            # This will make ~max_episode iterations, depending on effectively the lenght of
            # each episode
            for batch_observations in batch_rollouts_observations:

                reconstruction, mu, log_sigma = vision(batch_observations)
                loss = vision.loss(reconstruction, batch_observations, mu, log_sigma)
                optimizer.zero_grad()
                loss.backward()
                # This loss is now relative to batch_size elements
                optimizer.step()
                train_loss += loss.item()
        train_loss /= len(train_dataloader)
        return train_loss

    def _test_step(self, vision: ConvVAE, test_dataloader: RolloutDataloader):
        vision.eval()
        test_loss = 0
        for batch_rollouts_observations, _, _ in test_dataloader:
            # Move to the desired device, make device agnostic
            batch_rollouts_observations = batch_rollouts_observations.to(
                next(vision.parameters()).device
            )

            # Make a long tensor containing all the observations
            batch_rollouts_observations = batch_rollouts_observations.permute(
                1, 0, 2, 3, 4
            )

            # "break" the information relative to the sequentiality and shuffle

            shuffled_indices = torch.randperm(batch_rollouts_observations.shape[0])
            batch_rollouts_observations = batch_rollouts_observations[shuffled_indices]
            # Make the resulting shuffled observations a tensor of shape
            # number of elements in an episode x batch_size x observation shape
            batch_rollouts_observations = batch_rollouts_observations.permute(
                1, 0, 2, 3, 4
            )
            # This will make ~max_episode iterations, depending on effectively the lenght of
            # each episode
            for batch_observations in batch_rollouts_observations:
                reconstruction, mu, log_sigma = vision(batch_observations)
                loss = vision.loss(reconstruction, batch_observations, mu, log_sigma)
                test_loss += loss.item()
        test_loss /= len(test_dataloader)
        return test_loss

    def train(
        self,
        vision: ConvVAE,
        train_dataloader: RolloutDataloader,
        test_dataloader: RolloutDataloader,
        optimizer: torch.optim.Optimizer,
        epochs: int = 10,
        save_path=Path("models") / "vision.pt",
    ):
        vision.to(DEVICE)

        for epoch in range(epochs):
            train_loss = self._train_step(vision, train_dataloader, optimizer)
            test_loss = self._test_step(vision, test_dataloader)
            print(f"Epoch {epoch} | {train_loss=} | {test_loss=}")
        os.makedirs(save_path.parents[0], exist_ok=True)
        torch.save(vision.state_dict(), save_path)


# if __name__ == "__main__":
#     file_path = Path("data") / "dataset.pt"

#     if file_path.exists():
#         dataset = RolloutDataset.load(file_path=file_path)
#     else:
#         dataset = RolloutDataset(num_rollouts=10, max_steps=10)
#         dataset.save(file_path=file_path)

#     train_rollouts, test_rollouts, eval_rollouts = torch.utils.data.random_split(
#         dataset, [0.5, 0.3, 0.2]
#     )
#     # train_rollouts = cast(Subset[RolloutDataset], train_rollouts)
#     # test_rollouts = cast(Subset[RolloutDataset], test_rollouts)
#     # eval_rollouts = cast(Subset[RolloutDataset], eval_rollouts)
#     training_set = RolloutDataset(rollouts=train_rollouts.dataset.rollouts)  # type: ignore
#     test_set = RolloutDataset(rollouts=test_rollouts.dataset.rollouts)  # type: ignore
#     eval_set = RolloutDataset(rollouts=eval_rollouts.dataset.rollouts)  # type: ignore

#     train_dataloader = RolloutDataloader(training_set, 32)
#     test_dataloader = RolloutDataloader(test_set, 32)

#     vision = ConvVAE().to(DEVICE)

#     vision_trainer = VisionTrainer()

#     vision_trainer.train(
#         vision,
#         train_dataloader,
#         test_dataloader,
#         torch.optim.Adam(vision.parameters()),
#     )
