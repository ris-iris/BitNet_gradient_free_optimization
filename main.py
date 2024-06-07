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

    # Add optimizer arguments to the parser
    parser.add_argument("--lr", default=1e-4, type=float, help="Optimizer parameter: learning rate")
    parser.add_argument("--beta1", default=0.9, type=float, help="Optimizer parameter: beta1 for adam")
    parser.add_argument("--beta2", default=0.98, type=float, help="Optimizer parameter: beta2 for adam")
    parser.add_argument("--weight_decay", default=0.2, type=float, help="Optimizer parameter: weight decay for adam")
    parser.add_argument("--warmup_steps", default=1024, type=int, help="Optimizer parameter: warmup steps for adam")
    parser.add_argument("--max_grad_norm", default=10, type=int, help="Optimizer parameter: max grad norm for adam")

    parser.add_argument("--population_size", default=50, type=int, help="Optimizer parameter: population size")
    parser.add_argument("--treshold", default=15, type=int, help="Optimizer parameter: selection treshold")
    parser.add_argument("--bin_mutation_prob", default=0.5, type=float, help="Optimizer parameter: probability of mutating binary value")
    parser.add_argument("--emb_mutation_scale", default=1, type=float, help="Optimizer parameter: mutation scale for notmal mutaion of embeddings")

    parser.add_argument("--initial_temp", default=50.0, type=float, help="Optimizer parameter: initial temperature for simulated annealing")
    parser.add_argument("--cooling_rate", default=0.99, type=float, help="Optimizer parameter: cooling rate for simulated annealing")
    parser.add_argument("--min_temp", default=1e-3, type=float, help="Optimizer parameter: min temperature for simulated annealing")

    # TODO: add description
    parser.add_argument("--random_vec", default=10, type=int, help="Optimizer parameter: for zero order method")
    parser.add_argument("--momentum", default=0.9, type=float, help="Optimizer parameter: for zero order method")
    parser.add_argument("--grad_mode", default="zeroth_order_rge", help="Optimizer parameter: for zero order method")
    parser.add_argument("--v_step", default=10.0, type=float, help="Optimizer parameter: for zero order method")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # limiting to one GPU

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function
    wandb.login()
    run(args)

