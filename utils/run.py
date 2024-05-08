import torch
import random
import numpy as np

from datasets.datasets import get_dataset
from utils.train import train
from model.transformer import BitNetTransformer
from datasets.addition import AdditionDataset
from utils.utils import get_model, get_optimizer


def run(args):
    """
    Main function to run the script. It parses command-line arguments, loads the dataset, splits it into training and validation sets,
    creates data loaders, sets the loss function, and performs cross-validation to find the best hyperparameters and optimizer.
    """
    # Convert string arguments to appropriate data types
    dataset = args.dataset
    optimizer_name = args.optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = args.seed
    batch_size = args.batch_size
    epochs = args.epochs
    model_name = args.model
    data_repo = args.data_repo
    max_length = args.max_length

    print(f"Dataset: {dataset}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Using {device}")
    print(f"Seed: {seed}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Model: {model_name}")
    print(f"Max length: {args.max_length}")
    print(f"Data repo: {args.data_repo}")

    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataset, test_dataset, vocab_size = get_dataset(dataset, data_repo, max_length)

    model = get_model(model_name, vocab_size, max_length, len(train_dataset.label_to_id) if hasattr(train_dataset, 'label_to_id') else None)
    optimizer = get_optimizer(optimizer_name, model, torch.nn.functional.cross_entropy)



    train(train_dataset, test_dataset, optimizer, device, batch_size, epochs,
          model_save_root='models/', tensorboard_path="./tensorboard/part1_lr{}".format(0.001))
