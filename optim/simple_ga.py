import torch
from torch import nn

from optim.oprimizer import Optimizer

class SimpleGA(Optimizer):
    def __init__(self, model, loss_fn, population_size=50, treshold=15, bin_mutation_prob=0.5, emb_mutation_scale=1, **kwargs) -> None:
        super().__init__(model, loss_fn)
        self.model.eval()

        self.state_shape_dict = {k: v.shape for k, v in model.state_dict().items()}

        self.treshold = treshold
        self.population_size = population_size
        self.parents_idx = torch.arange(population_size)
        self.population = []
        self.__init_population()

        self.bin_mutation_prob = bin_mutation_prob
        self.emb_mutation = torch.randn
        self.emb_mutation_scale = emb_mutation_scale

    def __init_population(self):
        for i in range(self.population_size):
            state_dict = {}
            for layer, shape in self.state_shape_dict.items():
                if 'emb' in layer or 'to_logits' in layer or 'linear' in layer or 'weight_scale' in layer:
                    state_dict[layer] = torch.randn(shape)
                else:
                    state_dict[layer] = torch.bernoulli(0.5 * torch.ones(shape)) * 2 - 1
            self.population.append(state_dict)

    def __eval(self, input_ids, labels):
        with torch.no_grad():
            model = self.model
            losses = []
            for p in self.population:
                model.load_state_dict(p)
                outputs = model.forward(input_ids)
                if len(outputs.shape) == 3:
                    outputs = outputs.transpose(1, 2)
                losses.append(self.loss_fn(outputs, labels))
            losses = torch.tensor(losses)
            assert len(losses.shape) == 1, 'to many dimensions'
            self.model.load_state_dict(self.population[torch.argmin(losses)])
            self.parents_idx = torch.argsort(losses)[:self.treshold]
            return losses.min()
        
    def __mutate(self):
        new_population = []
        parent_choise = torch.randint(len(self.parents_idx), (self.population_size-1, ))
        for idx in parent_choise:
            state_dict = self.population[idx].copy()
            for layer, shape in self.state_shape_dict.items():
                if 'emb' in layer or 'to_logits' in layer or 'linear' in layer or 'weight_scale' in layer:
                    state_dict[layer] += self.emb_mutation(shape) * self.emb_mutation_scale
                else:
                    state_dict[layer] = torch.where(torch.rand(shape) < self.bin_mutation_prob, -state_dict[layer], state_dict[layer])
            new_population.append(state_dict)

        # saving the best from the previous iteration
        new_population.append(self.population[self.parents_idx[0]])
        self.population = new_population

    def step(self, input_ids, labels, track_ops=False):
        self.__mutate()
        loss = self.__eval(input_ids, labels)
        if track_ops:
            return loss, self.op_per_step(input_ids.shape[0], input_ids.shape[1])
        return loss
    
    def op_per_step(self, batch_size, seq_length):
        return {
            'float MACs forward': self.population_size * self.model.num_float_MACs(seq_length) * batch_size,
            'float MACs backward': 0,
            'int MACs': self.population_size * self.model.num_int_MACs(seq_length) * batch_size,
            'random numbers': (self.population_size - 1) * self.model.num_params()
        }