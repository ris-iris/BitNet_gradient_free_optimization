import torch
from torch import nn
from transformers import get_constant_schedule_with_warmup

from optim.oprimizer import Optimizer

class Adam(Optimizer):
    def __init__(self, model, loss_fn, max_grad_norm, lr=1e-4, betas=(0.9, 0.98), weight_decay=0.2, warmup_steps=-1) -> None:
        super().__init__(model, loss_fn)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        if warmup_steps > 0:
            self.scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps)
        else:
            self.scheduler = None
        
        self.max_grad_norm = max_grad_norm

        self.model.zero_grad()
        self.model.train()


    def step(self, input_ids, labels):
        # clear the gradients of all optimized parameters
        self.model.zero_grad()

        # get model's single-batch outputs and loss
        outputs = self.model.forward(input_ids)
        if len(outputs.shape) == 3:
            outputs = outputs.transpose(1, 2)
        loss = self.loss_fn(outputs, labels)

        # conduct back-proporgation
        loss.backward()

        # truncate gradient to max_grad_norm
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # step forward optimizer and scheduler
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        return loss