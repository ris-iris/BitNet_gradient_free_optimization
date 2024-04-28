import torch
from torch import nn

from optim.oprimizer import Optimizer

class SimpleGA(Optimizer):
    def __init__(self, model, loss_fn, population_size=50, treshold=5, bin_mutation_prob=0.1, emb_mutation=torch.randn, emb_mutation_scale = 0.1) -> None:
        super().__init__(model, loss_fn)
        self.model.eval()

        self.state_shape_dict = {k: v.shape for k, v in model.state_dict().items()}

        self.treshold = treshold
        self.population_size = population_size
        self.population = []
        self.__init_population()

        self.bin_mutation_prob = bin_mutation_prob
        self.emb_mutation = emb_mutation
        self.emb_mutation_scale = emb_mutation_scale

    def __init_population(self):
        for i in range(self.population_size):
            state_dict = {}
            for layer, shape in self.state_shape_dict.items():
                if 'emb' in layer:
                    state_dict[layer] = torch.randn(shape)
                else:
                    state_dict[layer] = torch.bernoulli(0.5 * torch.ones(shape))
            self.population.append(state_dict)

    def __eval(self, input_ids, labels):
        with torch.no_grad():
            model = self.model
            losses = []
            for p in self.population:
                model.load_state_dict(p)
                outputs = self.model.forward(input_ids)
                losses.append(self.loss_fn(outputs.transpose(1, 2), labels))
            losses = torch.tensor(losses)
            assert len(losses.shape) == 1, 'to many dimensions'
            return torch.argsort(losses)[:self.treshold], losses.mean()
        
    def __mutate(self, parents_idx):
        new_population = []
        parent_choise = torch.randint(len(parents_idx), (self.population_size, ))
        for idx in parent_choise:
            state_dict = self.population[idx]
            for layer, shape in self.state_shape_dict.items():
                if 'emb' in layer:
                    state_dict[layer] += self.emb_mutation(shape) * self.emb_mutation_scale
                else:
                    state_dict[layer] = torch.where(torch.rand(shape) < self.bin_mutation_prob, 1 - state_dict[layer], state_dict[layer])
            new_population.append(state_dict)
        self.population = new_population

    def step(self, input_ids, labels):
        parents_idx, loss = self.__eval(input_ids, labels)
        self.__mutate(parents_idx)
        return loss