# implementation from https://github.com/kyegomez/BitNet/tree/main
import torch

from torch import nn, Tensor
import torch.nn.functional as F


def activation_quant(x: Tensor):
    """Per token quantization to 8bits. No grouping is needed for quantization

    Args:
        x (Tensor): an activation tensor
    Returns:
        y (Tensor): a quantized activation tensor
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def activation_norm_quant(x: Tensor):
    """Per token quantization to 8bits. No grouping is needed for quantization

    Args:
        x (Tensor): _description_

    Returns:
        y (Tensor): a quantized activation tensor
        scale: a scalar for dequantization
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127)
    return y, scale


def weight_quant(w: Tensor):
    scale = w.abs().mean()
    e = w.mean()
    u = (w - e).sign() * scale
    return u


def weight_norm_quant(w: Tensor):
    scale = w.abs().mean()
    e = w.mean()
    u = (w - e).sign()
    return nn.Parameter(u), 1.0 / scale


def l2norm(t, dim=-1):
    return F.normalize(t, dim=dim)


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) module.

    Args:
        dim (int): The input dimension.
        affine (bool, optional): If True, apply an affine transformation to the normalized output.
            Default is True.

    Attributes:
        scale (float): The scaling factor for the normalized output.
        gamma (torch.Tensor or float): The learnable parameter for the affine transformation.

    """

    def __init__(self, dim, affine=True):
        super().__init__()
        self.scale = dim ** 0.5
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gamma = nn.Parameter(torch.ones(dim, device=device)) if affine else 1.0

    def forward(self, x):
        return l2norm(x) * self.gamma * self.scale


class BitLinear(nn.Linear):
    """
    Custom linear layer with bit quantization.

    Args:
        dim (int): The input dimension of the layer.
        training (bool, optional): Whether the layer is in training mode or not. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension of the layer.

    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

        # parameter that we will have to train as well as weights
        self.weight_scale = 1.0

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BitLinear layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        w = self.weight
        x_norm = RMSNorm(self.in_features)(x)

        if self.training:
            # STE using detach
            x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
            w_quant = w + (weight_quant(w) - w).detach()
            y = F.linear(x_quant, w_quant)
        else:
            w_scale = self.weight_scale
            x_quant, x_scale = activation_norm_quant(x_norm)

            # TODO:replace F.linear with low dimension kernel
            y = F.linear(x_quant, w) / w_scale / x_scale
        return y

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        if not self.training and mode:
            raise RuntimeError("only transition from train to eval mode allowed")
        return super().train(mode)

    def eval(self):
        if not (self.training):
            return super().eval()

        # weight quantization
        self.weight, self.weight_scale = weight_norm_quant(self.weight)

        return super().eval()
