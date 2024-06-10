from typing import Callable, Optional
import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from model.bit_linear import BitLinear, RMSNorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BitFeedForward(nn.Module):
    """
    BitFeedForward module performs feed-forward operations on the input tensor.

    Args:
        dim (int): The input dimension.
        dim_out (int, optional): The output dimension. If not provided, it is set to the input dimension.
        mult (int, optional): The multiplier for the inner dimension. Default is 4.
        swish (bool, optional): Whether to use Swish activation. Default is False.
        post_act_ln (bool, optional): Whether to apply Layer Normalization after activation. Default is False.
        dropout (float, optional): The dropout probability. Default is 0.0.
        no_bias (bool, optional): Whether to exclude bias in linear layers. Default is False.
    """

    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            mult: int = 4,
            swish: bool = False,
            post_act_ln: bool = False,
            dropout: float = 0.0,
            no_bias: bool = False,
            *args,
            **kwargs
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out else dim

        if swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        self.ff = nn.Sequential(
            BitLinear(dim, inner_dim, bias=not no_bias, *args, **kwargs),
            activation,
            nn.LayerNorm(inner_dim) if post_act_ln else None,
            nn.Dropout(dropout),
            BitLinear(inner_dim, dim_out, bias=not no_bias, *args, **kwargs),
        )

    def forward(self, x):
        return self.ff(x)
    
    def eval(self):
        for mod in self.ff:
            mod.eval()
        return super().eval()


class BitMHA(nn.Module):
    # slightly changed implementation from https://medium.com/pytorch/training-compact-transformers-from-scratch-in-30-minutes-with-pytorch-ff5c21668ed5
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "Embedding dim must be divisible by number of heads."
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = BitLinear(embed_dim, embed_dim * 3)
        self.projection = BitLinear(embed_dim, embed_dim)

    def forward(self, x, is_causal=True):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)  # B, N, (3*C)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)  # B, N, 3(qkv), H(eads), embed_dim
            .permute(2, 0, 3, 1, 4)  # 3, B, H(eads), N, emb_dim
        )
        q, k, v = torch.chunk(qkv, 3)  # B, H, N, dim

        # B,H,N,dim x B,H,dim,N -> B,H,N,N
        attn = torch.bmm(q.reshape(-1, N, C // self.num_heads),
                         k.reshape(-1, N, C // self.num_heads).transpose(-2, -1)) * self.scale  # <q,k> / sqrt(d)
        if is_causal:
            mask = torch.ones((N, N)).tril_().unsqueeze(0).to(device)
            attn = attn.masked_fill(mask == 0, -9e15)
        attn = attn.softmax(dim=-1)  # Softmax over embedding dim

        x = (  # B, H, N, N
            torch.bmm(attn, v.reshape(-1, N, C // self.num_heads))  # B,H,N,N x B,H,N,dim -> B, H, N, dim
            .reshape(B, self.num_heads, N, C // self.num_heads)
            .transpose(1, 2)  # B, N, H, dim
            .reshape(B, N, C)  # B, N, (H*dim)
        )
        x = self.projection(x)

        return x
    
    def eval(self):
        self.qkv.eval()
        self.projection.eval()
        return super().eval()


class Transformer(nn.Module):
    """
    Transformer module that applies multi-head attention and feed-forward layers.

    Args:
        dim (int): The dimension of the input and output tensors.
        heads (int): The number of attention heads.
        depth (int): The number of transformer layers.
        ff_mult (int, optional): The multiplier for the hidden dimension in the feed-forward layers.
            Defaults to 2.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        layers (nn.ModuleList): List of multi-head attention layers.
        ffn_layers (nn.ModuleList): List of feed-forward layers.

    """

    def __init__(
            self, dim: int, heads: int, depth: int, ff_mult: int = 2, *args, **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(BitMHA(dim, heads, *args, **kwargs))

            self.ffn_layers.append(
                BitFeedForward(
                    dim,
                    dim,
                    ff_mult,
                    swish=True,
                    post_act_ln=True,
                    dropout=0.1,
                ),
            )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        skip = x
        for attn, ffn in zip(self.layers, self.ffn_layers):
            x = attn(x, is_causal=True, *args, **kwargs)
            x = x + skip
            x = ffn(x) + x
        return x
    
    def eval(self):
        for attn, ffn in zip(self.layers, self.ffn_layers):
            attn.eval()
            ffn.eval()
        return super().eval()

# From pytorch docs
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        return x + self.pe[:x.size(0)]
    
# [MAIN MODEL] BitNetTransformer
class BitNetTransformer(nn.Module):
    """
    BitNetTransformer is a transformer-based model for BitNet.

    Args:
        dim (int): The dimension of the token embeddings.
        depth (int): The number of transformer layers.
        num_tokens (int): The number of tokens in the vocabulary.
        heads (int, optional): The number of attention heads in the transformer. Defaults to 8.
        ff_mult (int, optional): The multiplier for the feed-forward layer dimension. Defaults to 4.

    Examples:
    >>> import torch
    >>> from bitnet import BitNetTransformer
    >>> x = torch.randint(0, 20000, (1, 1024))
    >>> bitnet = BitNetTransformer(
    ...     num_tokens=20000,
    ...     dim=1024,
    ...     depth=6,
    ...     heads=8,
    ...     ff_mult=4,
    ... )
    >>> logits = bitnet(x)
    >>> print(logits)
    """

    def __init__(
            self,
            dim: int,
            depth: int,
            num_tokens: int,
            heads=2,
            ff_mult=1,
            output_dim: int = None,
            max_length=128
    ):
        super().__init__()

        self.emb = nn.Embedding(num_tokens, dim)
        self.pe = PositionalEncoding(dim, max_length)

        self.transformer = Transformer(
            dim=dim, depth=depth, heads=heads, ff_mult=ff_mult
        )
        if output_dim is None:
            output_dim = num_tokens
        self.to_logits = nn.Sequential(RMSNorm(dim), nn.Linear(dim, output_dim))

        # calculate MACs in one forward pass
        self.transformer_macs = depth * (
            4 * dim * dim # MH Attention MACs
            + 2 * dim * dim * ff_mult # feedforward MACs
        )
        self.attention_macs = depth * dim
        self.logits_macs = dim * output_dim

    def forward(self, x):
        x = self.emb(x)
        x = self.pe(x)
        x = self.transformer(x)
        return self.to_logits(x)
    
    def num_float_MACs(self, seq_length):
        if self.training:
            return (self.logits_macs + self.transformer_macs) * seq_length +  self.attention_macs * seq_length * seq_length
        else:
            return self.logits_macs * seq_length +  self.attention_macs * seq_length * seq_length

    def num_int_MACs(self, seq_length):
        if self.training:
            return 0
        else:
            return self.transformer_macs * seq_length

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def eval(self):
        self.emb.eval()
        self.transformer.eval()
        self.to_logits.eval()
        return super().eval()
