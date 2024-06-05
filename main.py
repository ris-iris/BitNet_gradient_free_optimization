import argparse
import os

import wandb

# Import custom modules
from utils.run import run


# Main function
if __name__ == "__main__":
    """
    Main function
    """
    # Create argument parser
    parser = argparse.ArgumentParser()
    # Add arguments to the parser
    parser.add_argument("--dataset", default="brackets", help="The dataset to use for training and testing.")
    parser.add_argument("--batch_size", default=30, type=int, help="The batch size for training.")
    parser.add_argument("--epochs", default=4, type=int, help="The number of epochs to train for.")
    parser.add_argument("--optimizer", default="adam", help="The optimizer to use for training.")
    parser.add_argument("--seed", default=42, type=int, help="The random seed for reproducibility.")
    parser.add_argument("--model", default="bit_sa_transformer", help="The model architecture to use for training.")
    parser.add_argument("--max_length", default=128, type=int, help="The maximum sequence length for the model.")
    parser.add_argument("--data_repo", default="./data/", help="The directory where the dataset is stored.")
    parser.add_argument("--track_ops", default=False, type=bool, help="The flag that specifies if operations will be tracked.")

    # os.environ["WANDB_DISABLED"] = "true"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # limiting to one GPU

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function
    wandb.login()
    run(args)

