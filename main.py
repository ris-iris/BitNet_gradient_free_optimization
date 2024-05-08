import torch
import argparse
import os

from torch.utils.data import DataLoader
from transformers import BertTokenizer

from datasets.datasets import SADataset
from model.sa import SATransformer
# Import custom modules
from utils.run import run

from utils.train import train
from model.transformer import BitNetTransformer
from datasets.addition import AdditionDataset
from optim.adam import Adam
from optim.simple_ga import SimpleGA
from optim.mcmc import MCMC


if __name__ == "not__main__":
        # print('Training with MCMC')
        # batch_size, epochs = -1, 20
        # model = BitNetTransformer(64, 2, 15, 2)
        # optimizer = MCMC(model, torch.nn.functional.cross_entropy)
        # train_dataset, test_dataset = AdditionDataset(1024), AdditionDataset(128)
        #
        # train(train_dataset, test_dataset, optimizer, 'cpu', batch_size, epochs,
        #         model_save_root='models/', tensorboard_path="./tensorboard/part1_lr{}".format(0.001))
        #
        # print('Training with Genetic Algorithm')
        # batch_size, epochs = -1, 5
        # model = BitNetTransformer(64, 2, 15, 2)
        # optimizer = SimpleGA(model, torch.nn.functional.cross_entropy)
        # train_dataset, test_dataset = AdditionDataset(1024), AdditionDataset(128)
        #
        # train(train_dataset, test_dataset, optimizer, 'cpu', batch_size, epochs,
        #         model_save_root='models/', tensorboard_path="./tensorboard/part1_lr{}".format(0.001))
        #
        # print('Training with Adam')
        # batch_size, epochs = 4, 20
        # model = BitNetTransformer(64, 2, 15, 2)
        # optimizer = Adam(model, torch.nn.functional.cross_entropy, max_grad_norm=10, lr=1e-4, betas=(0.9, 0.98), weight_decay=0.2, warmup_steps=1024)
        # train_dataset, test_dataset = AdditionDataset(1024), AdditionDataset(128)
        #
        # train(train_dataset, test_dataset, optimizer, 'cpu', batch_size, epochs,
        #         model_save_root='models/', tensorboard_path="./tensorboard/part1_lr{}".format(0.001))

        batch_size, epochs = 30, 1
        train_data_repo = './data/twitter_training.json'
        test_data_repo = './data/twitter_validation.json'
        labels = ['Positive', 'Neutral', 'Negative', 'Irrelevant']
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        max_length = 130
        train_dataset = SADataset(train_data_repo, tokenizer, labels)
        test_dataset = SADataset(test_data_repo, tokenizer, labels)
        # model = BitNetTransformer(max_length, 3, tokenizer.vocab_size, 2, 4)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = SATransformer(dim=64, depth=2, num_tokens=tokenizer.vocab_size, transformer_output_dim=2, output_dim=4, max_length=max_length).to(device)
        optimizer = Adam(model, torch.nn.functional.cross_entropy, max_grad_norm=10, lr=1e-4, betas=(0.9, 0.98), weight_decay=0.2, warmup_steps=1024)
        train(train_dataset, test_dataset, optimizer, device, batch_size, epochs,
                model_save_root='models/', tensorboard_path="./tensorboard/part1_lr{}".format(0.001))


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
    parser.add_argument("--data_repo", default="./data", help="The directory where the dataset is stored.")

    os.environ["WANDB_DISABLED"] = "true"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # limiting to one GPU

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function
    run(args)

