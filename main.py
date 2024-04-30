import torch
import argparse
import os

# Import custom modules
from utils.run import run

from utils.train import train
from model.transformer import BitNetTransformer
from datasets.addition import AdditionDataset
from optim.adam import Adam
from optim.simple_ga import SimpleGA
from optim.mcmc import MCMC


if __name__ == "__main__":
        print('Training with MCMC')
        batch_size, epochs = -1, 20
        model = BitNetTransformer(64, 2, 15, 2)
        optimizer = MCMC(model, torch.nn.functional.cross_entropy)
        train_dataset, test_dataset = AdditionDataset(1024), AdditionDataset(128)

        train(train_dataset, test_dataset, optimizer, 'cpu', batch_size, epochs,
                model_save_root='models/', tensorboard_path="./tensorboard/part1_lr{}".format(0.001))
        
        print('Training with Genetic Algorithm')
        batch_size, epochs = -1, 5
        model = BitNetTransformer(64, 2, 15, 2)
        optimizer = SimpleGA(model, torch.nn.functional.cross_entropy)
        train_dataset, test_dataset = AdditionDataset(1024), AdditionDataset(128)

        train(train_dataset, test_dataset, optimizer, 'cpu', batch_size, epochs,
                model_save_root='models/', tensorboard_path="./tensorboard/part1_lr{}".format(0.001))
        
        print('Training with Adam')
        batch_size, epochs = 4, 20
        model = BitNetTransformer(64, 2, 15, 2)
        optimizer = Adam(model, torch.nn.functional.cross_entropy, max_grad_norm=10, lr=1e-4, betas=(0.9, 0.98), weight_decay=0.2, warmup_steps=1024)
        train_dataset, test_dataset = AdditionDataset(1024), AdditionDataset(128)

        train(train_dataset, test_dataset, optimizer, 'cpu', batch_size, epochs,
                model_save_root='models/', tensorboard_path="./tensorboard/part1_lr{}".format(0.001))
        

# Main function
if __name__ == "not__main__":
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

