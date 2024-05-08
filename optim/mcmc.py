import torch
from torch import nn

from optim.oprimizer import Optimizer

class MCMC(Optimizer):
    def __init__(self, model, loss_fn, bin_mutation_prob=0.5, emb_mutation=torch.randn, emb_mutation_scale=1) -> None:
        super().__init__(model, loss_fn)
        self.temp_f = -1
        self.temp_loss = -1

        self.bin_mutation_prob = bin_mutation_prob
        self.emb_mutation = emb_mutation
        self.emb_mutation_scale = emb_mutation_scale

    def __new_model(self):
        state_dict = self.model.state_dict().copy()
        for layer in state_dict.keys():
            shape = state_dict[layer].shape
            if 'emb' in layer:
                state_dict[layer] += self.emb_mutation(shape).to(self.model.device) * self.emb_mutation_scale
            else:
                state_dict[layer] = torch.where(torch.rand(shape) < self.bin_mutation_prob, 1 - state_dict[layer], state_dict[layer])

        return state_dict
    
    def step(self, input_ids, labels):
        if self.temp_f == -1:
            outputs = self.model.forward(input_ids)
            if len(outputs.shape) == 3:
                outputs = outputs.transpose(1, 2)
            self.temp_loss = self.loss_fn(outputs, labels)
            self.temp_f = torch.exp(-self.temp_loss)

        old_state_dict = self.model.state_dict().copy()
        self.model.load_state_dict(self.__new_model())
        outputs = self.model.forward(input_ids)
        if len(outputs.shape) == 3:
            outputs = outputs.transpose(1, 2)
        loss = self.loss_fn(outputs, labels)
        new_f = torch.exp(-loss)

        if torch.rand(1).item() < new_f / self.temp_f:
            self.temp_f = new_f
            self.temp_loss = loss
        else:
            self.model.load_state_dict(old_state_dict)
        
        return self.temp_loss

        