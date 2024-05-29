import torch
from torch import nn
from optim.oprimizer import Optimizer
import copy
import random

class SimulatedAnnealing(Optimizer):
    def __init__(self, model, loss_fn, initial_temp=100.0, cooling_rate=0.99, min_temp=1e-3) -> None:
        super().__init__(model, loss_fn)
        self.current_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp

        # Save the initial model parameters as the best parameters
        self.best_params = copy.deepcopy(model.state_dict())
        self.best_loss = float('inf')

    def step(self, input_ids, labels):
        # Evaluate the current model's loss
        outputs = self.model(input_ids)
        if len(outputs.shape) == 3:
            outputs = outputs.transpose(1, 2)
        current_loss = self.loss_fn(outputs, labels)

        # Check if current loss is the best loss
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_params = copy.deepcopy(self.model.state_dict())

        # Generate new candidate parameters
        new_params = self.__generate_new_params()

        # Temporarily load new parameters
        old_params = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(new_params)

        # Evaluate new parameters
        outputs = self.model(input_ids)
        if len(outputs.shape) == 3:
            outputs = outputs.transpose(1, 2)
        new_loss = self.loss_fn(outputs, labels)

        # Acceptance probability calculation
        acceptance_prob = self.__acceptance_probability(current_loss, new_loss, self.current_temp)

        # Decide whether to accept the new parameters
        if new_loss < current_loss or random.random() < acceptance_prob:
            current_loss = new_loss
            self.best_params = new_params
            self.best_loss = new_loss
        else:
            # Revert to old parameters if new ones are not accepted
            self.model.load_state_dict(old_params)

        # Cool down the temperature
        self.current_temp = max(self.current_temp * self.cooling_rate, self.min_temp)

        return self.best_loss

    def __generate_new_params(self):
        # Create a new set of parameters with small random perturbations
        new_params = copy.deepcopy(self.model.state_dict())
        for param_name in new_params.keys():
            perturbation = torch.randn_like(new_params[param_name]) * self.current_temp
            new_params[param_name].add_(perturbation)
        return new_params

    def __acceptance_probability(self, current_loss, new_loss, temperature):
        if new_loss < current_loss:
            return 1.0
        else:
            return torch.exp((current_loss - new_loss) / temperature).item()
