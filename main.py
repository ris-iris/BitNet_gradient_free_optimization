# Import necessary libraries
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch
import argparse
import os

# Import custom modules
from utils import run

# Main function
if __name__ == "__main__":
    """
    Main function to run the script. It parses command-line arguments, loads the dataset, splits it into training and validation sets,
    creates data loaders, sets the loss function, and performs cross-validation to find the best hyperparameters and optimizer.
    """
    # Create argument parser
    parser = argparse.ArgumentParser()
    # Add arguments to the parser
    parser.add_argument("--dataset", default="MNIST", help="The dataset to use for training and testing.")
    parser.add_argument("--batch_size", default=100, type=int, help="The batch size for training.")
    parser.add_argument("--epochs", default=2, type=int, help="The number of epochs to train for.")
    parser.add_argument("--plot", action="store_true", help="Whether to plot the training and validation curves.")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate for the optimizer.")
    parser.add_argument("--optimizer", default="SGD", help="The optimizer to use for training.")
    parser.add_argument("--save", action="store_true", help="Whether to save the trained model.")
    parser.add_argument("--save_path", default="./results/",
                        help="The directory where the trained model should be saved.")
    parser.add_argument("--verbose", action="store_true", help="Whether to print detailed training progress.")
    parser.add_argument("--scheduler", action="store_true", help="Whether to use a learning rate scheduler.")
    parser.add_argument("--num_iter", default=5, type=int, help="The number of different models to train.")
    parser.add_argument("--seed", default=42, type=int, help="The random seed for reproducibility.")
    parser.add_argument("--model", default="bit_transformer", help="The model architecture to use for training.")
    parser.add_argument("--max_length", default=512, type=int, help="The maximum sequence length for the model.")
    parser.add_argument("--data_repo", default="./data/", help="The directory where the dataset is stored.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="The maximum gradient norm for clipping.")
    parser.add_argument("--warmup_percent", default=0.1, type=float, help="The percentage of warmup steps.")

    os.environ["WANDB_DISABLED"] = "true"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # limiting to one GPU

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function
    run(args)

