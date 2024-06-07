import torch
from copy import deepcopy

from optim.oprimizer import Optimizer
import torch.autograd.forward_ad as fwAD
from torch.func import functional_call


class ZAD(Optimizer):
    name = 'ZAD'

    def __init__(self, model, criterion, lr=1e-3, random_vec=10, momentum=0.9, grad_mode='zeroth_order_rge', v_step=1, mutation_prob=0.5, threshold=1e-3, **kwargs):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super(ZAD, self).__init__(model, criterion)
        self.model.eval()

        self.lr = lr
        self.random_vec = random_vec
        self.threshold = threshold
        self.f = None
        self.momentum = momentum

        self.params_dict = {name: p for name, p in self.model.named_parameters()}

        self.grad = [torch.zeros(p.size()).to(self.device) for p in self.params_dict.values()]
        self.params_data = [p.data for p in self.params_dict.values()]

        assert grad_mode in ['zeroth_order_rge', 'zeroth_order_cge']
        self.grad_mode = grad_mode
        self.v_step = v_step
        self.mutation_prob = mutation_prob

    def op_per_step(self, batch_size, seq_length):
        if self.grad_mode == 'zeroth_order_rge':
            return {
                'float MACs forward': self.model.num_float_MACs(seq_length) * batch_size * self.random_vec,
                'float MACs backward': 0,
                'int MACs': self.model.num_int_MACs(seq_length) * batch_size * self.random_vec,
                'random numbers': self.random_vec * self.model.num_params()
            }
        elif self.grad_mode == 'zeroth_order_cge':
            return {
                'float MACs forward': self.model.num_float_MACs(seq_length) * batch_size * self.model.num_params(),
                'float MACs backward': 0,
                'int MACs': self.model.num_int_MACs(seq_length) * batch_size * self.model.num_params(),
                'random numbers': 0
            }
        

    def step(self, input_ids, labels, track_ops=False):

        if self.grad_mode == 'zeroth_order_rge':
            # we do integer randomized gradient estimation (RGE) with projected SGD
            with torch.no_grad():
                torch._foreach_mul_(self.grad, self.momentum)
                loss = self.loss_fn(functional_call(self.model, self.params_dict, input_ids), labels)

                for _ in range(self.random_vec):
                    params_v = {}
                    v = []
                    for layer, param in self.params_dict.items():
                        if 'emb' in layer or 'to_logits' in layer or 'linear' in layer:
                            v.append(torch.randn(param.shape).to(self.device))
                            params_v[layer] = param + v[-1] * self.v_step
                        else:
                            params_v[layer] = torch.where(torch.rand(param.shape) < self.mutation_prob, 1 - param, param)
                            v.append(params_v[layer] - param)
                        
                    lossv = self.loss_fn(functional_call(self.model, params_v, input_ids), labels).item()
                    torch._foreach_mul_(v, (1 - self.momentum) * (lossv - loss.item()) / (self.random_vec * self.v_step))
                    torch._foreach_add_(self.grad, v)

                for layer, param, grad in zip(self.params_dict.keys(), self.params_data, self.grad):
                    if 'emb' in layer or 'to_logits' in layer or 'linear' in layer:
                        param -= self.lr * grad
                    else:
                        param = torch.where(torch.abs(self.lr * grad) >= 0.5, 1-param, param)


        elif self.grad_mode == 'zeroth_order_cge':
            # we do integer coordinate-wise gradient estimation (CGE) with projected SGD
            with torch.no_grad():
                torch._foreach_mul_(self.grad, self.momentum)
                params_v = deepcopy(self.params_dict)
                loss = self.loss_fn(functional_call(self.model, self.params_dict, input_ids), labels)
                for i, (key, param) in enumerate(self.params_dict.items()):
                    if 'emb' in layer or 'to_logits' in layer or 'linear' in layer:
                        for j in range(param.numel()):
                            if j != 0:
                                params_v[key].data.view(-1)[j-1] -= self.v_step
                            params_v[key].data.view(-1)[j] += self.v_step
                            loss_v = self.loss_fn(functional_call(self.model, params_v, input_ids), labels).item()
                            self.grad[i].view(-1)[j] += (1 - self.momentum) * (loss_v - loss.item()) / self.v_step
                        params_v[key].data.view(-1)[param.numel()-1] -= self.v_step
                    else:
                        for j in range(param.numel()):
                            v_step = -1 if params_v[key].data.view(-1)[j] == 1 else 1
                            if j != 0:
                                params_v[key].data.view(-1)[j-1] = 1 - params_v[key].data.view(-1)[j-1]
                            params_v[key].data.view(-1)[j] += v_step
                            loss_v = self.loss_fn(functional_call(self.model, params_v, input_ids), labels).item()
                            self.grad[i].view(-1)[j] += (1 - self.momentum) * (loss_v - loss.item()) / v_step
                        params_v[key].data.view(-1)[param.numel()-1] -= 1 - params_v[key].data.view(-1)[param.numel()-1]

                for layer, param, grad in zip(self.params_dict.keys(), self.params_data, self.grad):
                    if 'emb' in layer or 'to_logits' in layer or 'linear' in layer:
                        param -= self.lr * grad
                    else:
                        param = torch.where(torch.abs(self.lr * grad) >= 0.5, 1-param, param) 

        if track_ops:
            return loss, self.op_per_step(input_ids.shape[0], input_ids.shape[1])
        return loss


