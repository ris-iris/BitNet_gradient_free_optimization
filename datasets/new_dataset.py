from transformers import BertTokenizer
from typing import List
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import json
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, tokenizer: BertTokenizer, data: List[str], labels: List[int]):
        self.tokenizer = tokenizer
        self.data = data

        # 0 means "yes", 1 means "no"
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokenized = self.tokenizer(self.data[idx], add_special_tokens=True, return_tensors="pt")
        return {
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": self.labels[idx]
        }
    
def get_dataloader(tokenizer: BertTokenizer, data: List[str], labels: List[int], batch_size: int) -> DataLoader:
    dataset = Dataset(tokenizer, data, labels)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    return dataloader

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    with open("brackets_dataset.json", "r") as f:
        ds = json.load(f)
        data = ds["data"]
        labels = ds["labels"]

    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2)
    
    batch_size = 2

    data_train_dataloader = get_dataloader(tokenizer, data_train, labels_train, batch_size)
    data_test_dataloader = get_dataloader(tokenizer, data_test, labels_test, batch_size)
    for batch in data_train_dataloader:
        print(batch)
        break
