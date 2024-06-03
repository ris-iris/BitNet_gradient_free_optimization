from torch import nn

from model.transformer import Transformer, BitNetTransformer


class SATransformer(nn.Module):
    def __init__(
            self,
            dim: int,
            depth: int,
            num_tokens: int,
            transformer_output_dim: int,
            output_dim: int,
            heads=8,
            ff_mult=4,
            max_length=128
    ):
        super().__init__()
        self.transformer = BitNetTransformer(
            dim=dim, depth=depth, heads=heads, ff_mult=ff_mult, num_tokens=num_tokens, output_dim=transformer_output_dim
        )
        self.linear = nn.Linear(transformer_output_dim*max_length, output_dim)

        # calculate MACs in one forward pass
        self.linear_macs = transformer_output_dim*max_length * output_dim

    def forward(self, x):
        x = self.transformer(x)
        x = x.flatten(1)
        x = self.linear(x)
        return x

    def num_float_MACs(self, seq_length):
        return self.transformer.num_float_MACs(seq_length) + self.linear_macs

    def num_int_MACs(self, seq_length):
        return self.transformer.num_int_MACs(seq_length)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())