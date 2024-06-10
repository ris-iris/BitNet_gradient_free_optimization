import numpy as np

from model.sa import SATransformer
from model.transformer import BitNetTransformer
from optim.adam import Adam
from optim.mcmc import MCMC
from optim.simple_ga import SimpleGA
from optim.simulated_annealing import SimulatedAnnealing
from optim.zeroth import ZAD


from transformers import BertTokenizer

from datasets.addition import AdditionDataset
from datasets.bracket import BracketTokenizer, BracketDataset
from datasets.sa import SADataset


def compute_metrics(predictions, gold_labels):
    """
    Compute evaluation metrics (confusion matrix and F1 scores) for SA task.

    INPUT: 
      - gold_labels: real labels;
      - predictions: model predictions.
    OUTPUT:
      - confusion matrix;
      - f1 scores for positive and negative classes.
    """

    confusion_matrix = np.zeros((2, 2))
    for pred, gold in zip(predictions, gold_labels):
        confusion_matrix[gold][pred] += 1

    tp = confusion_matrix[0][0]
    fp = confusion_matrix[1][0]
    fn = confusion_matrix[0][1]
    tn = confusion_matrix[1][1]
    f1_positive = 2 * tp / (2 * tp + fp + fn)
    f1_negative = 2 * tn / (2 * tn + fp + fn)

    return confusion_matrix, f1_positive, f1_negative


def get_model(model_name, vocab_size, max_length=None, output_dim=None):
    """
    Get the model for the specified task.
    """
    if model_name == "bit_transformer":
        return BitNetTransformer(64, 2, vocab_size, 2)
    elif model_name == "bit_sa_transformer":
        return SATransformer(dim=64, depth=2, num_tokens=vocab_size, transformer_output_dim=output_dim, output_dim=output_dim, max_length=max_length)
    else:
        raise ValueError(f"Model {model_name} not supported")

def get_optimizer(optimizer_name, model, loss_fn, opt_kwargs):
    """
    Get the optimizer for the specified task.
    """
    if optimizer_name == "adam":
        return Adam(model, loss_fn, **opt_kwargs)
    elif optimizer_name == "simple_ga":
        return SimpleGA(model, loss_fn, **opt_kwargs)
    elif optimizer_name == "mcmc":
        return MCMC(model, loss_fn, **opt_kwargs)
    elif optimizer_name == "sim_annealing":
        return SimulatedAnnealing(model, loss_fn, **opt_kwargs)
    elif optimizer_name == "zeroth":
        return ZAD(model, loss_fn, **opt_kwargs)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")

def get_dataset(dataset_name, data_repo=None, max_length=128):
    """
    Get the dataset for training and evaluation.

    INPUT:
      - dataset_name: the name of the dataset
      - data_repo: the path to the dataset
      - max_length: the maximum length of the sentence

    OUTPUT:
      - train_dataset: the training dataset
      - test_dataset: the test dataset
      - vocab_size: the size of the vocabulary
    """
    if dataset_name == "twitter":
        labels = ['Positive', 'Neutral', 'Negative', 'Irrelevant']
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        train_dataset = SADataset(data_repo + "/twitter_training.json", tokenizer, labels, max_length)
        test_dataset = SADataset(data_repo + "/twitter_validation.json", tokenizer, labels, max_length)
        vocab_size = tokenizer.vocab_size
    elif dataset_name == "addition":
        train_dataset, test_dataset = AdditionDataset(1024), AdditionDataset(128)
        vocab_size = 15
    elif dataset_name == "brackets":
        tokenizer = BracketTokenizer()
        train_dataset = BracketDataset(data_repo + "/train_brackets_dataset.json", tokenizer, max_length)
        test_dataset = BracketDataset(data_repo + "/test_brackets_dataset.json", tokenizer, max_length)
        vocab_size = len(tokenizer.vocab)
    else:
        raise ValueError("Invalid dataset name.")

    return train_dataset, test_dataset, vocab_size

def get_tokenizer(model_name):
    """
    Get the tokenizer for the specified model.
    """
    # TODO: Implement this function
    # from transformers import AutoTokenizer
    # return AutoTokenizer.from_pretrained(model_name)
    pass
