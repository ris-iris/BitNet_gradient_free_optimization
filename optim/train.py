from tqdm import tqdm
import jsonlines

import torch
import torch.utils.data
from torch import nn, optim
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_constant_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter


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


def train(train_dataset, dev_dataset, model, device, batch_size, epochs,
          learning_rate, warmup_percent, max_grad_norm, model_save_root,
          tensorboard_path="./tensorboard"):
    '''
    Train models with predefined datasets.

    INPUT:
      - train_dataset: dataset for training
      - dev_dataset: dataset for evlauation
      - model: model to train
      - device: hardware device for training ('cpu' or 'cuda')
      - batch_size: batch size for load the dataset
      - epochs: total epochs to train the model
      - learning_rate: learning rate of optimizer
      - warmup_percent: percentage of warmup steps
      - max_grad_norm: maximum gradient for clipping
      - model_save_root: path to save model checkpoints
    '''

    tb_writer = SummaryWriter(tensorboard_path)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        collate_fn=train_dataset.collate_fn
    )

    # calculate total training steps
    total_steps = epochs * len(train_dataloader)
    warmup_steps = int(total_steps * warmup_percent)

    # set up optimizer and constant learning rate scheduler with warmup
    # TODO
    # optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), weight_decay=0.2)
    optimizer = ...
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

    model.zero_grad()
    model.train()
    best_dev_macro_f1 = 0
    total_train_step = 0
    log_freq = 20
    save_repo = model_save_root + 'lr{}-warmup{}'.format(learning_rate, warmup_percent)

    for epoch in range(epochs):

        train_loss_accum = 0.0
        epoch_train_step = 0
        running_loss = 0.0

        for i, batch in enumerate(tqdm(train_dataloader, desc="Training")):

            # clear the gradients of all optimized parameters
            model.zero_grad()

            epoch_train_step += 1
            total_train_step += 1

            batch_tuple = tuple(input_tensor.to(device) for input_tensor in batch)
            input_ids, labels = batch_tuple

            # get model's single-batch outputs and loss
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            # conduct back-proporgation
            loss.backward()

            # truncate gradient to max_grad_norm
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            train_loss_accum += loss.mean().item()

            running_loss += loss.mean().item()
            if i > 0 and i % log_freq == 0:
                # print(f'[Step {total_train_step:5d}] loss: {running_loss / log_freq:.3f}')
                tb_writer.add_scalar("task1_roberta/loss/train", running_loss / log_freq, total_train_step)
                running_loss = 0.0

            # step forward optimizer and scheduler
            optimizer.step()
            scheduler.step()

        epoch_train_loss = train_loss_accum / epoch_train_step

        # epoch evaluation
        dev_loss, confusion, f1_pos, f1_neg = evaluate(dev_dataset, model, device, batch_size)
        macro_f1 = (f1_pos + f1_neg) / 2
        tb_writer.add_scalar("training/loss/eval", dev_loss, total_train_step)

        print(f'Epoch: {epoch} | Training Loss: {epoch_train_loss:.3f} | Validation Loss: {dev_loss:.3f}')
        print(f'Epoch {epoch} Validation:')
        print(f'Confusion Matrix:')
        print(confusion)
        print(f'F1: ({f1_pos * 100:.2f}%, {f1_neg * 100:.2f}%) | Macro-F1: {macro_f1 * 100:.2f}%')

        if macro_f1 > best_dev_macro_f1:
            best_dev_macro_f1 = macro_f1
            model.save_pretrained(save_repo)
            train_dataset.tokenizer.save_pretrained(save_repo)
            print("Model Saved!")

    tb_writer.flush()
    tb_writer.close()


def evaluate(eval_dataset, model, device, batch_size, use_labels=True, result_save_file=None):
    '''
    Evaluate the trained model.

    INPUT: 
      - eval_dataset: dataset for evaluation
      - model: trained model
      - device: hardware device for training ('cpu' or 'cuda')
      - batch_size: batch size for load the dataset
      - use_labels: whether the gold labels should be used as one input to the model
      - result_save_file: path to save the prediction results
    '''
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=batch_size,
        collate_fn=eval_dataset.collate_fn
    )

    eval_loss_accum = 0
    eval_step = 0
    batch_preds = []
    batch_labels = []

    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluation"):

        eval_step += 1

        with torch.no_grad():
            batch_tuple = tuple(input_tensor.to(device) for input_tensor in batch)
            input_ids, labels = batch_tuple

            if use_labels:
                outputs = model(input_ids, labels=labels)
            else:
                outputs = model(input_ids)
            loss = outputs.loss
            logits = outputs.logits

            batch_preds.append(logits.detach().cpu().numpy())
            if use_labels:
                batch_labels.append(labels.detach().cpu().numpy())
                eval_loss_accum += loss.mean().item()

    pred_labels = np.argmax(np.concatenate(batch_preds), axis=1)

    if result_save_file:
        pred_results = eval_dataset.decode_class(pred_labels)
        with jsonlines.open(result_save_file, mode="w") as writer:
            for sid, pred in enumerate(pred_results):
                sample = eval_dataset.get_text_sample(sid)
                sample["prediction"] = pred
                writer.write(sample)

    if use_labels:
        eval_loss = eval_loss_accum / eval_step
        gold_labels = list(np.concatenate(batch_labels))
        confusion, f1_pos, f1_neg = compute_metrics(pred_labels, gold_labels)
        return eval_loss, confusion, f1_pos, f1_neg
    else:
        return None
