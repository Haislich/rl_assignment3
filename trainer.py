import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from controller import Controller, ControllerTrainer
from latent_dataset import LatentDataloader, LatentDataset
from memory import MDN_RNN, MemoryTrainer
from rollout_dataset import Episode, RolloutDataloader, RolloutDataset
from vision import ConvVAE, VisionTrainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_dataset_gif(
    episode: Episode,
    save_path=Path("media/rollout_dataset.gif"),
):
    observations = episode.observations.unsqueeze(0).to(DEVICE)
    scale_factor = 1
    img_width, img_height = 64 * scale_factor, 64 * scale_factor
    total_width = img_width
    total_height = img_height
    images = []
    for t in range(observations.shape[1]):
        original_img = T.Resize((img_height, img_width))(
            T.ToPILImage()(observations[0, t].cpu())
        )
        combined_img = Image.new("RGB", (total_width, total_height), (0, 0, 0))
        combined_img.paste(original_img, (0, 0))
        images.append(combined_img)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Save as GIF
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=len(images) / 60,
        loop=0,
    )
    print(f"Dataset GIF saved to {save_path}")


def create_vision_gif(
    episode: Episode,
    vision: ConvVAE,
    save_path=Path("media/vision_reconstruction.gif"),
):
    observations = episode.observations.unsqueeze(0).to(DEVICE)
    latents = vision.get_batched_latents(observations)
    vae_reconstructions = vision.decoder(latents.squeeze(0))
    scale_factor = 1
    spacing = 1
    img_width, img_height = 64 * scale_factor, 64 * scale_factor
    total_width = img_width * 2 + spacing * 2
    total_height = img_height

    images = []
    for t in range(vae_reconstructions.shape[0]):
        original_img = T.Resize((img_height, img_width))(
            T.ToPILImage()(observations[0, t].cpu())
        )
        vae_img = T.Resize((img_height, img_width))(
            T.ToPILImage()(vae_reconstructions[t].cpu())
        )
        combined_img = Image.new("RGB", (total_width, total_height), (0, 0, 0))
        combined_img.paste(original_img, (0, 0))
        combined_img.paste(vae_img, (img_width + spacing, 0))
        images.append(combined_img)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Save as GIF
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=len(images) / 60,
        loop=0,
    )
    print(f"Vision reconstruction GIF saved to {save_path}")


def create_memory_gif(
    episode: Episode,
    vision: ConvVAE,
    memory: MDN_RNN,
    save_path=Path("media/vision_memory_reconstruction.gif"),
):
    observations = episode.observations.unsqueeze(0).to(DEVICE)
    actions = episode.actions.unsqueeze(0).to(DEVICE)

    # Get latent representations from VAE
    latents = vision.get_batched_latents(observations)

    # Initialize RNN hidden state
    hidden_state, cell_state = memory.init_hidden()
    hidden_state = hidden_state.to(DEVICE)
    cell_state = cell_state.to(DEVICE)

    # Generate predictions using MDN-RNN
    predicted_latents = []
    for t in range(latents.shape[1] - 1):
        pi, mu, sigma, hidden_state, cell_state = memory(
            latents[:, t, :], actions[:, t, :], None, None
        )
        predicted_latent = memory.sample_latent(pi, mu, sigma)
        predicted_latents.append(predicted_latent)

    predicted_latents = torch.stack(predicted_latents, dim=1)

    # Decode the latents
    vae_reconstructions = vision.decoder(latents.squeeze(0))
    memory_reconstructions = vision.decoder(predicted_latents.squeeze(0))

    # Set up visualization parameters
    scale_factor = 1
    spacing = 1
    img_width, img_height = 64 * scale_factor, 64 * scale_factor
    total_width = img_width * 3 + spacing * 3
    total_height = img_height

    images = []

    for t in range(vae_reconstructions.shape[0] - 1):
        original_img = T.Resize((img_height, img_width))(
            T.ToPILImage()(observations[0, t].cpu())
        )
        vision_img = T.Resize((img_height, img_width))(
            T.ToPILImage()(vae_reconstructions[t].cpu())
        )
        memory_img = T.Resize((img_height, img_width))(
            T.ToPILImage()(memory_reconstructions[t].cpu())
        )

        combined_img = Image.new("RGB", (total_width, total_height), (0, 0, 0))
        combined_img.paste(original_img, (0, 0))
        combined_img.paste(vision_img, (img_width + spacing, 0))
        combined_img.paste(memory_img, (2 * (img_width + spacing), 0))
        images.append(combined_img)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as GIF
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=len(images) / 60,
        loop=0,
    )
    print(f"Vision and Memory reconstruction GIF saved to {save_path}")


def create_rollout_dataset(
    continuous=True, num_rollouts: int = 10000, max_steps: int = 500
):
    rollout_dataset = RolloutDataset(num_rollouts, max_steps, continuous=continuous)
    episode = Episode.load(rollout_dataset.episodes_paths[0])
    create_dataset_gif(episode)


def train_vision(
    epochs: int = 1,
    batch_size=64,
    continuous=True,
    num_rollouts: int = 10000,
    max_steps: int = 500,
):
    rollout_dataset = RolloutDataset(num_rollouts, max_steps, continuous=continuous)

    (
        train_episodes,
        test_episodes,
        val_episodes,
    ) = torch.utils.data.random_split(rollout_dataset, [0.7, 0.2, 0.1])
    train_dataset = RolloutDataset.from_subset(train_episodes)
    test_dataset = RolloutDataset.from_subset(test_episodes)
    val_dataset = RolloutDataset.from_subset(val_episodes)
    train_dataloader = RolloutDataloader(train_dataset, batch_size)
    test_dataloader = RolloutDataloader(test_dataset, batch_size)
    val_dataloader = RolloutDataloader(val_dataset, batch_size)
    vision = ConvVAE().to(DEVICE)
    vision_trainer = VisionTrainer(vision)
    vision_trainer.train(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        optimizer=torch.optim.Adam(vision.parameters()),
    )
    episode = Episode.load(rollout_dataset.episodes_paths[0])
    create_vision_gif(episode, vision)


def create_latent_dataset(
    continuous=True, num_rollouts: int = 10000, max_steps: int = 500
):
    rollout_dataset = RolloutDataset(num_rollouts, max_steps, continuous=continuous)
    vision = ConvVAE.from_pretrained()
    LatentDataset(rollout_dataset=rollout_dataset, vision=vision)


def train_memory(
    epochs=10,
    batch_size: int = 64,
    continuous=True,
    num_rollouts: int = 10000,
    max_steps: int = 500,
):
    rollout_dataset = RolloutDataset(num_rollouts, max_steps, continuous=continuous)
    (
        train_episodes,
        test_episodes,
        val_episodes,
    ) = torch.utils.data.random_split(rollout_dataset, [0.7, 0.2, 0.1])

    train_dataset = RolloutDataset.from_subset(train_episodes)
    test_dataset = RolloutDataset.from_subset(test_episodes)
    val_dataset = RolloutDataset.from_subset(val_episodes)

    vision = ConvVAE.from_pretrained().to(DEVICE)
    latent_training_set = LatentDataset(train_dataset, vision)
    latent_test_set = LatentDataset(test_dataset, vision)
    latent_val_set = LatentDataset(val_dataset, vision)

    train_dataloader = LatentDataloader(latent_training_set, batch_size)
    test_dataloader = LatentDataloader(latent_test_set, batch_size)
    val_dataloader = LatentDataloader(latent_val_set, batch_size)

    memory = MDN_RNN(continuous=continuous).to(DEVICE)
    memory_trainer = MemoryTrainer(memory)
    memory_trainer.train(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        val_dataloader=val_dataloader,
        optimizer=torch.optim.Adam(memory.parameters()),
        epochs=epochs,
    )
    episode = Episode.load(rollout_dataset.episodes_paths[0])
    create_memory_gif(episode, vision, memory)


def train_controller(
    population_size=32, max_iterations=1, continuous=True, render=False
):

    vision = ConvVAE.from_pretrained().to(DEVICE)
    memory = MDN_RNN.from_pretrained().to(DEVICE)
    controller = Controller(continuous=continuous)
    controller_trainer = ControllerTrainer(
        controller, vision, memory, population_size=population_size, render=render
    )
    controller_trainer.train(max_iterations)


def main():
    parser = argparse.ArgumentParser(description="Training script")

    # Subcommands for each function
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Parser for create_rollout_dataset
    parser_create_rollout = subparsers.add_parser(
        "create_rollout_dataset", help="Create a rollout dataset"
    )
    parser_create_rollout.add_argument(
        "--continuous",
        type=bool,
        default=True,
        help="Whether the rollout is continuous",
    )
    parser_create_rollout.add_argument(
        "--num_rollouts", type=int, default=10000, help="Number of rollouts"
    )
    parser_create_rollout.add_argument(
        "--max_steps", type=int, default=500, help="Maximum steps per rollout"
    )

    # Parser for train_vision
    parser_train_vision = subparsers.add_parser(
        "train_vision", help="Train the vision model"
    )
    parser_train_vision.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser_train_vision.add_argument(
        "--batch_size", type=int, default=64, help="Number of training epochs"
    )
    parser_train_vision.add_argument(
        "--continuous",
        type=bool,
        default=True,
        help="Whether the rollout is continuous",
    )
    parser_train_vision.add_argument(
        "--num_rollouts", type=int, default=10000, help="Number of rollouts"
    )
    parser_train_vision.add_argument(
        "--max_steps", type=int, default=500, help="Maximum steps per rollout"
    )

    # Parser for create_latent
    parser_create_latent = subparsers.add_parser(
        "create_latent_dataset", help="Create a latent dataset"
    )
    parser_create_latent.add_argument(
        "--continuous",
        type=bool,
        default=True,
        help="Whether the rollout is continuous",
    )
    parser_create_latent.add_argument(
        "--num_rollouts", type=int, default=10000, help="Number of rollouts"
    )
    parser_create_latent.add_argument(
        "--max_steps", type=int, default=500, help="Maximum steps per rollout"
    )

    # Parser for train_memory
    parser_train_memory = subparsers.add_parser(
        "train_memory", help="Train the memory model"
    )
    parser_train_memory.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser_train_memory.add_argument(
        "--batch_size", type=int, default=64, help="Batch size"
    )
    parser_train_memory.add_argument(
        "--continuous",
        type=bool,
        default=True,
        help="Whether the rollout is continuous",
    )
    parser_train_memory.add_argument(
        "--num_rollouts", type=int, default=10000, help="Number of rollouts"
    )
    parser_train_memory.add_argument(
        "--max_steps", type=int, default=500, help="Maximum steps per rollout"
    )

    # Parser for train_controller
    parser_train_controller = subparsers.add_parser(
        "train_controller", help="Train the controller"
    )
    parser_train_controller.add_argument(
        "--population_size", type=int, default=32, help="Population size for training"
    )
    parser_train_controller.add_argument(
        "--max_iterations", type=int, default=1, help="Maximum number of iterations"
    )
    parser_train_controller.add_argument(
        "--continuous",
        type=bool,
        default=True,
        help="Whether the rollout is continuous",
    )
    parser_train_controller.add_argument(
        "--render", type=bool, default=False, help="Render the rollouts"
    )

    args = parser.parse_args()

    # Execute the appropriate function
    if args.command == "create_rollout_dataset":
        create_rollout_dataset(args.continuous, args.num_rollouts, args.max_steps)
    elif args.command == "train_vision":
        train_vision(
            args.epochs,
            args.batch_size,
            args.continuous,
            args.num_rollouts,
            args.max_steps,
        )
    elif args.command == "create_latent_dataset":
        create_latent_dataset(args.continuous, args.num_rollouts, args.max_steps)
    elif args.command == "train_memory":
        train_memory(
            args.epochs,
            args.batch_size,
            args.continuous,
            args.num_rollouts,
            args.max_steps,
        )
    elif args.command == "train_controller":
        train_controller(
            args.population_size, args.max_iterations, args.continuous, args.render
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
