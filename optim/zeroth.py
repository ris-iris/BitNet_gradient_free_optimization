import torch
from copy import deepcopy

from optim.oprimizer import Optimizer
import torch.autograd.forward_ad as fwAD
from torch.func import functional_call


class ZAD(Optimizer):
    name = 'ZAD'

    def __init__(self, model, criterion, random_vec=10, momentum=0.9, names=None, grad_mode='zeroth_order_rge',
                 v_step=1, threshold=1e-3, device='cpu'):
        self.device = device
        super(ZAD, self).__init__(model, criterion)
        # self.lr = lr
        self.random_vec = random_vec
        self.threshold = threshold
        self.f = None
        self.momentum = momentum
        self.grad = [torch.zeros(p.size()).to(self.device) for p in self.model.parameters()]
        self.params = [p for p in self.model.parameters()]
        self.params_data = [p.data for p in self.params]
        self.names = names
        assert grad_mode in ['zeroth_order_rge', 'zeroth_order_cge']
        self.grad_mode = grad_mode
        self.params_dict = {name: p for name, p in zip(self.names, self.params)}
        self.v_step = v_step

    def op_per_step(self, batch_size, seq_length):
        if self.grad_mode == 'zeroth_order_forward-mode_AD':
            return {
                'float MACs forward': 0,
                'float MACs backward': 0,
                'int MACs': 0,
                'random numbers': 0
            }
        elif self.grad_mode == 'zeroth_order_rge':
            return {
                'float MACs forward': 0,
                'float MACs backward': 0,
                'int MACs': 0,
                'random numbers': (self.random_vec - 1) * self.model.num_params()
            }
        elif self.grad_mode == 'zeroth_order_cge':
            return {
                'float MACs forward': 0,
                'float MACs backward': 0,
                'int MACs': self.model.num_params(),
                'random numbers': 0
            }
    def step(self, input_ids, labels, track_ops=False):

        if self.grad_mode == 'zeroth_order_rge':
            # we do integer randomized gradient estimation (RGE) with signSGD
            with torch.no_grad():
                torch._foreach_mul_(self.grad, self.momentum)
                loss = self.loss_fn(functional_call(self.model, self.params_dict, input_ids), labels).item()

                total_v = [torch.zeros(p.size(), dtype=torch.int64).to(self.device) for p in self.params_data]
                for _ in range(self.random_vec):
                    v = [torch.randint(0, 2, p.size()).to(self.device) - 1 for p in self.params_data]
                    params_v = deepcopy(self.params_dict)
                    for p, v_ in zip(params_v.items(), v):
                        p[1].data += v_ * self.v_step
                    lossv = self.loss_fn(functional_call(self.model, params_v, input_ids), labels).item()
                    torch._foreach_mul_(v, (lossv - loss))
                    torch._foreach_add_(total_v, v)

                torch._foreach_mul_(total_v, (1 - self.momentum))
                torch._foreach_add_(self.grad, total_v)
                grad_with_threshold = [torch.where(torch.abs(g) > self.threshold, g, torch.zeros_like(g, dtype=torch.int8)) for g in self.grad]
                torch._foreach_sign_(grad_with_threshold)
                torch._foreach_add_(self.params_data, grad_with_threshold)
                # we clip the weights to 0 and 1
                torch._foreach_clamp_(self.params_data, 0, 1)


        elif self.grad_mode == 'zeroth_order_cge':
            # we do integer coordinate-wise gradient estimation (CGE) with signSGD
            with torch.no_grad():
                torch._foreach_mul_(self.grad, self.momentum)
                params_v = deepcopy(self.params_dict)
                loss = self.loss_fn(functional_call(self.model, self.params_dict, input_ids), labels).item()
                for i, (key, param) in enumerate(self.params_dict.items()):
                    total_v = torch.zeros(param.size(), dtype=torch.int64).to(self.device)
                    for j in range(param.numel()):
                        if j != 0:
                            params_v[key].data.view(-1)[j-1] -= self.v_step
                        params_v[key].data.view(-1)[j] += self.v_step
                        loss_v = self.loss_fn(functional_call(self.model, params_v, input_ids), labels).item()
                        total_v.view(-1)[j] += (loss_v - loss)

                    self.grad[i].view(-1)[j] += (1 - self.momentum) * total_v
                    params_v[key].data.view(-1)[param.numel()-1] -= self.v_step

                grad_with_threshold = [
                    torch.where(torch.abs(g) > self.threshold, g, torch.zeros_like(g, dtype=torch.int8)) for g in
                    self.grad]
                torch._foreach_sign_(grad_with_threshold)
                torch._foreach_add_(self.params_data, grad_with_threshold)
                # we clip the weights to 0 and 1
                torch._foreach_clamp_(self.params_data, 0, 1)
        if track_ops:
            return loss, self.op_per_step(input_ids.shape[0], input_ids.shape[1])
        return loss


