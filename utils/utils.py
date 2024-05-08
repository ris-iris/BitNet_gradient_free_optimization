import numpy as np

from model.sa import SATransformer
from model.transformer import BitNetTransformer
from optim.adam import Adam
from optim.mcmc import MCMC
from optim.simple_ga import SimpleGA


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

def get_optimizer(optimizer_name, model, loss_fn):
    """
    Get the optimizer for the specified task.
    """
    if optimizer_name == "adam":
        return Adam(model, loss_fn, lr=1e-4, betas=(0.9, 0.98), weight_decay=0.2, warmup_steps=1024)
    elif optimizer_name == "simple_ga":
        return SimpleGA(model, loss_fn)
    elif optimizer_name == "mcmc":
        return MCMC(model, loss_fn)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")

def get_tokenizer(model_name):
    """
    Get the tokenizer for the specified model.
    """
    # TODO: Implement this function
    # from transformers import AutoTokenizer
    # return AutoTokenizer.from_pretrained(model_name)
    pass
