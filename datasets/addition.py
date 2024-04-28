import torch
from torch.utils.data import Dataset
import numpy as np

class AdditionDataset(Dataset):
    def __init__(self, size) -> None:
        self.size = size
        self.problems = self.__generate_problems__()

    def __generate_problems__(self):
        def str_to_labels(s):
            def int_or_label(c):
                if c.isdigit():
                    return int(c)
                elif c == '+':
                    return 10
                return 11
            return torch.tensor([12, ] + [int_or_label(c) for c in s] + [13, ])

        add = torch.randint(0, 1000, (self.size, 3))
        results = add.sum(dim=1)
        problems = ['+'.join([str(term.item()) for term in terms]) + '=' + str(res.item()) for terms, res in zip(add, results)]
        return [str_to_labels(p) for p in problems]
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        problem = self.problems[idx]
        return problem[:-1], problem[1:]
    
    def collate_fn(self, data):
        def pad_labels(t, max_length):
            result = torch.ones(max_length - t.shape[0], dtype=torch.int) * 14
            return torch.cat((t, result))
        
        max_length = np.max([l[1].shape[0] for l in data])
        collated_one_hot = torch.stack([pad_labels(o[0], max_length) for o in data])
        collated_labels = torch.stack([pad_labels(l[1], max_length) for l in data])
        return collated_one_hot, collated_labels