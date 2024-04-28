import numpy as np


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


def get_tokenizer(model_name):
    """
    Get the tokenizer for the specified model.
    """
    # TODO: Implement this function
    # from transformers import AutoTokenizer
    # return AutoTokenizer.from_pretrained(model_name)
    pass
