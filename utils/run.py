import torch
import random
import numpy as np

from utils.train import train
from model.transformer import BitNetTransformer
from datasets.addition import AdditionDataset

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

    # tokenizer = get_tokenizer(model_name)
    # TODO
    model = BitNetTransformer(64, 2, 12, 2)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.2)
    train_dataset, test_dataset = AdditionDataset(1024), AdditionDataset(128)

    # Create data loaders for the training, validation, and test sets

    train(train_dataset, test_dataset, model, optimizer, device, batch_size, epochs, lr, warmup_percent, max_grad_norm,
          model_save_root='models/', tensorboard_path="./tensorboard/part1_lr{}".format(lr))
