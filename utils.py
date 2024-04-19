import random

import numpy as np
import torch
from datasets import get_dataset
from optim import optimizers
from optim.train import train


def get_tokenizer(model_name):
    """
    Get the tokenizer for the specified model.
    """
    # TODO: Implement this function
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


def run(args):
    """
    Main function to run the script. It parses command-line arguments, loads the dataset, splits it into training and validation sets,
    creates data loaders, sets the loss function, and performs cross-validation to find the best hyperparameters and optimizer.
    """
    # Convert string arguments to appropriate data types
    dataset = args.dataset
    lr = args.lr
    optimizer_name = args.optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = args.seed
    batch_size = args.batch_size
    epochs = args.epochs
    max_grad_norm = args.max_grad_norm
    warmup_percent = args.warmup_percent
    model_name = args.model
    data_repo = args.data_repo
    max_length = args.max_length

    print(f"Dataset: {dataset}")
    print(f"Learning rate: {lr}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Using {device}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Seed: {seed}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Model: {model_name}")
    print(f"Max length: {args.max_length}")
    print(f"Data repo: {args.data_repo}")
    print(f"Max grad norm: {max_grad_norm}")
    print(f"Warmup percent: {warmup_percent}")

    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tokenizer = get_tokenizer(model_name)
    # TODO
    model = ...
    model.to(device)
    optimizer = getattr(optimizers, optimizer_name)(model.parameters(), lr=lr)

    train_dataset, test_dataset, labels, input_dim = get_dataset(dataset, data_repo=data_repo, tokenizer=tokenizer,
                                                                 max_length=max_length)

    # Create data loaders for the training, validation, and test sets

    train(train_dataset, test_dataset, model, device, batch_size, epochs, lr, warmup_percent, max_grad_norm,
          model_save_root='models/', tensorboard_path="./tensorboard/part1_lr{}".format(lr))
