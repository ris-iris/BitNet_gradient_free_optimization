import torch
import random
import numpy as np
import wandb

from utils.train import train
from utils.utils import get_model, get_optimizer, get_dataset


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
    track_ops = args.track_ops

    config = {
        "Dataset": dataset, 
        "Optimizer": optimizer_name, 
        "Using device": device,
        "Seed": seed,
        "Batch size": batch_size,
        "Epochs": epochs,
        "Model": model_name,
        "Max length": max_length,
        "Data repo": data_repo,
        "Tracking OPs": track_ops,
    }

    print(config)

    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataset, test_dataset, vocab_size = get_dataset(dataset, data_repo, max_length)

    model = get_model(model_name, vocab_size, max_length+2, len(train_dataset.label_to_id) if hasattr(train_dataset, 'label_to_id') else None)
    model.to(device)
    optimizer = get_optimizer(optimizer_name, model, torch.nn.functional.cross_entropy)

    wandb.init(project="bitNet_gradient_free", group=dataset + "_dataset", config=config)

    train(train_dataset, test_dataset, optimizer, device, batch_size, epochs, track_ops=track_ops)
    
    wandb.finish()
