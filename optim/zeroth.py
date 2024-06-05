import torch
from copy import deepcopy

from optim.oprimizer import Optimizer
import torch.autograd.forward_ad as fwAD
from torch.func import functional_call


class ZAD(Optimizer):
    name = 'ZAD'

    def __init__(self, model, criterion, lr=0.001, random_vec=10, momentum=0.9, names=None, grad_mode='zeroth_order_rge',
                 v_step=10.0, device='cpu'):
        self.device = device
        super(ZAD, self).__init__(model, criterion)
        self.lr = lr
        self.random_vec = random_vec
        self.f = None
        self.momentum = momentum
        self.grad = [torch.zeros(p.size()).to(self.device) for p in self.model.parameters()]
        self.params = [p for p in self.model.parameters()]
        self.params_data = [p.data for p in self.params]
        self.names = names
        assert grad_mode in ['zeroth_order_rge', 'zeroth_order_forward-mode_AD', 'zeroth_order_cge']
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

        if self.grad_mode == 'zeroth_order_forward-mode_AD':
            with torch.no_grad():
                torch._foreach_mul_(self.grad, self.momentum)
                total_loss = 0.0
                for _ in range(self.random_vec):
                    tangents = {name: torch.clip(torch.rand_like(p), min=1e-3) for name, p in self.params_dict.items()}
                    v = [t for t in tangents.values()]

                    dual_params = {}
                    with fwAD.dual_level():
                        for name, p in self.params_dict.items():
                            dual_params[name] = fwAD.make_dual(p, tangents[name])
                        loss = self.loss_fn(functional_call(self.model, dual_params, input_ids), labels)
                        jvp_result = fwAD.unpack_dual(loss).tangent
                    torch._foreach_mul_(v, jvp_result.item() * (1 - self.momentum) / self.random_vec)
                    torch._foreach_add_(self.grad, v)
                    total_loss += loss.item()
                torch._foreach_add_(self.params_data, torch._foreach_mul(self.grad, -self.lr))
                loss = total_loss / self.random_vec

        elif self.grad_mode == 'zeroth_order_rge':
            with torch.no_grad():
                torch._foreach_mul_(self.grad, self.momentum)
                loss = self.loss_fn(functional_call(self.model, self.params_dict, input_ids), labels).item()

                for _ in range(self.random_vec):
                    v = [torch.randn(p.size()).to(self.device) for p in self.params_data]
                    params_v = deepcopy(self.params_dict)
                    for p, v_ in zip(params_v.items(), v):
                        p[1].data += v_ * self.v_step
                    lossv = self.loss_fn(functional_call(self.model, params_v, input_ids), labels).item()
                    torch._foreach_mul_(v, (1 - self.momentum) * (lossv - loss) / (self.random_vec * self.v_step))
                    torch._foreach_add_(self.grad, v)

                torch._foreach_add_(self.params_data, torch._foreach_mul(self.grad, -self.lr))

        elif self.grad_mode == 'zeroth_order_cge':
            with torch.no_grad():
                torch._foreach_mul_(self.grad, self.momentum)
                params_v = deepcopy(self.params_dict)
                loss = self.loss_fn(functional_call(self.model, self.params_dict, input_ids), labels).item()
                for i, (key, param) in enumerate(self.params_dict.items()):
                    for j in range(param.numel()):
                        if j != 0:
                            params_v[key].data.view(-1)[j-1] -= self.v_step
                        params_v[key].data.view(-1)[j] += self.v_step
                        loss_v = self.loss_fn(functional_call(self.model, params_v, input_ids), labels).item()
                        self.grad[i].view(-1)[j] += (1 - self.momentum) * (loss_v - loss) / self.v_step
                    params_v[key].data.view(-1)[param.numel()-1] -= self.v_step

                torch._foreach_add_(self.params_data, torch._foreach_mul(self.grad, -self.lr))

        if track_ops:
            return loss, self.op_per_step(input_ids.shape[0], input_ids.shape[1])
        return loss


