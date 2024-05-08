from copy import deepcopy
from torch.utils.data import Dataset
from tqdm import tqdm
import jsonlines

import torch
import torch.utils.data
import numpy as np
from transformers import BertTokenizer

from datasets.addition import AdditionDataset


class SADataset(Dataset):
    """
    SADataset in Pytorch
    """

    def __init__(self, data_repo, tokenizer, labels, sent_max_length=128):

        self.label_to_id = {label: i for i, label in enumerate(labels)}
        self.id_to_label = {i: label for i, label in enumerate(labels)}

        self.tokenizer = tokenizer

        self.pad_token = self.tokenizer.pad_token
        self.pad_id = self.tokenizer.pad_token_id

        self.text_samples = []
        self.samples = []

        print("Building Dataset...")

        with jsonlines.open(data_repo, "r") as reader:
            for sample in tqdm(reader.iter()):
                # print(sample)
                self.text_samples.append(sample)
                # TODO
                input_ids = self.tokenizer.encode(sample['input'], max_length=sent_max_length, truncation=True,
                                                  add_special_tokens=False)
                input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
                special_tokens_count = self.tokenizer.num_special_tokens_to_add()
                input_ids = self.padding([input_ids], max_length=sent_max_length + special_tokens_count)[0]
                label_id = self.label_to_id[sample['label']]

                self.samples.append({"ids": input_ids, "label": label_id})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return deepcopy(self.samples[index])

    def padding(self, inputs, max_length=-1):
        """
        Pad inputs to the max_length.

        INPUT: 
          - inputs: input token ids
          - max_length: the maximum length you should add padding to.

        OUTPUT: 
          - pad_inputs: token ids padded to `max_length` """

        if max_length < 0:
            max_length = np.max([len(input_ids) for input_ids in inputs])

        pad_inputs = [input_ids + [self.pad_id] * (max_length - len(input_ids)) for input_ids in inputs]

        return pad_inputs

    def collate_fn(self, batch):
        """
        Convert batch inputs to tensor of batch_ids and labels.

        INPUT: 
          - batch: batch input, with format List[Dict1{"ids":..., "label":...}, Dict2{...}, ..., DictN{...}]

        OUTPUT: 
          - tensor_batch_ids: torch tensor of token ids of a batch, with format Tensor(List[ids1, ids2, ..., idsN])
          - tensor_labels: torch tensor for corresponding labels, with format Tensor(List[label1, label2, ..., labelN])
        """

        tensor_batch_ids = torch.tensor([sample["ids"] for sample in batch])
        tensor_labels = torch.tensor([sample["label"] for sample in batch])

        return tensor_batch_ids, tensor_labels

    def get_text_sample(self, index):
        return deepcopy(self.text_samples[index])

    def decode_class(self, class_ids):
        """
        Decode to output the predicted class name.

        INPUT: 
          - class_ids: index of each class.

        OUTPUT: 
          - labels_from_ids: a list of label names. """

        label_name_list = [self.id_to_label[class_id] if class_id in self.id_to_label else "unknown" for class_id in
                           class_ids]

        return label_name_list


# TODO: Implement the get_dataset function
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
        pass


    return train_dataset, test_dataset, vocab_size
