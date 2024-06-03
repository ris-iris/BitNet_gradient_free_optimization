from tqdm import tqdm

import torch
import torch.utils.data
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# from torch.utils.tensorboard import SummaryWriter


def train(train_dataset, dev_dataset, optimizer, device, batch_size, epochs, model_save_root,
          track_ops=False):
    '''
    Train models with predefined datasets.

    INPUT:
      - train_dataset: dataset for training
      - dev_dataset: dataset for evlauation
      - optimizer: optimizer to use
      - device: hardware device for training ('cpu' or 'cuda')
      - batch_size: batch size for load the dataset
      - epochs: total epochs to train the model
      - model_save_root: path to save model checkpoints
    '''

    # tb_writer = SummaryWriter(tensorboard_path)
    if batch_size < 0:
        batch_size = len(train_dataset)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        collate_fn=train_dataset.collate_fn
    )

    total_train_step = 0
    log_freq = 20
    # save_repo = model_save_root + 'lr{}-warmup{}'.format(learning_rate, warmup_percent)

    for epoch in range(epochs):

        train_loss_accum = 0.0
        epoch_train_step = 0
        running_loss = 0.0

        for i, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            epoch_train_step += 1
            total_train_step += 1

            batch_tuple = tuple(input_tensor.to(device) for input_tensor in batch)
            input_ids, labels = batch_tuple

            if track_ops:
                loss, ops = optimizer.step(input_ids, labels, True)
            else:
                loss = optimizer.step(input_ids, labels)

            train_loss_accum += loss.mean().item()

            running_loss += loss.mean().item() # why mean?
            if i > 0 and i % log_freq == 0:
                #tb_writer.add_scalar("task1_roberta/loss/train", running_loss / log_freq, total_train_step)
                running_loss = 0.0

        epoch_train_loss = train_loss_accum / epoch_train_step

        # epoch evaluation
        dev_loss = evaluate(dev_dataset, optimizer.model, optimizer.loss_fn, device, batch_size)
        #tb_writer.add_scalar("training/loss/eval", dev_loss, total_train_step)

        print(f'Epoch: {epoch} | Training Loss: {epoch_train_loss:.3f} | Validation Loss: {dev_loss:.3f}')
        
    #tb_writer.flush()
    #tb_writer.close()


def evaluate(eval_dataset, model, loss_fn, device, batch_size):
    '''
    Evaluate the trained model.

    INPUT: 
      - eval_dataset: dataset for evaluation
      - model: trained model
      - loss_fn: loss function
      - device: hardware device for training ('cpu' or 'cuda')
      - batch_size: batch size for load the dataset
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

    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluation"):

        eval_step += 1

        with torch.no_grad():
            batch_tuple = tuple(input_tensor.to(device) for input_tensor in batch)
            input_ids, labels = batch_tuple

            outputs = model(input_ids)
            if len(outputs.shape) == 3:
                outputs = outputs.transpose(1, 2)
            loss = loss_fn(outputs, labels)

            eval_loss_accum += loss.mean().item()

    return eval_loss_accum / eval_step
