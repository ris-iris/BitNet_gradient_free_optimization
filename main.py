import torch

from model.bit_linear import BitLinear
from model.transformer import Transformer, BitNetTransformer

print('_____BitLinear_______')
linear = BitLinear(256, 128)
x = torch.randn((1, 256))
y = linear(x)
print(y)
print(y.shape)

print('_____BitLinear__eval______')
linear.eval()
y = linear(x)
print(y)
print(y.shape)

print('_____Transformer_______')
transformer = Transformer(256, 4, 4)
transformer.eval()
x = torch.randn((1, 10, 256))
y = transformer(x)
print(y)
print(y.shape)
