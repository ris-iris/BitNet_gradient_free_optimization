import torch
from torch import nn

from optim.oprimizer import Optimizer

class MCMC(Optimizer):
    def __init__(self, model, loss_fn, bin_mutation_prob=0.5, emb_mutation_scale=1, **kwargs) -> None:
        super().__init__(model, loss_fn)
        self.model.eval()
        
        self.temp_f = -1
        self.temp_loss = -1

        self.bin_mutation_prob = bin_mutation_prob
        self.emb_mutation = torch.randn
        self.emb_mutation_scale = emb_mutation_scale

    def __new_model(self):
        state_dict = self.model.state_dict().copy()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for layer in state_dict.keys():
            shape = state_dict[layer].shape
            if 'emb' in layer or 'to_logits' in layer or 'linear' in layer or 'weight_scale' in layer:
                state_dict[layer] += self.emb_mutation(shape).to(device) * self.emb_mutation_scale
            else:
                state_dict[layer] = torch.where(torch.rand(shape).to(device) < self.bin_mutation_prob, -state_dict[layer], state_dict[layer])

        return state_dict
    
    def step(self, input_ids, labels, track_ops=False):
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

        if track_ops:
            return self.temp_loss, self.op_per_step(input_ids.shape[0], input_ids.shape[1])
        return self.temp_loss

    def op_per_step(self, batch_size, seq_length):
        return {
            'float MACs forward': self.model.num_float_MACs(seq_length) * batch_size,
            'float MACs backward': 0,
            'int MACs': self.model.num_int_MACs(seq_length) * batch_size,
            'random numbers': self.model.num_params()
        }