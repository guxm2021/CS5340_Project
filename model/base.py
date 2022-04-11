import math
import torch
import torch.nn as nn
from typing import Optional


# utility classes for Transformer
class PositionalEncoding(nn.Module):
    """This class implements the absolute sinusoidal positional encoding function.

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))

    Arguments
    ---------
    input_size: int
        Embedding dimension.
    max_len : int, optional
        Max length of the input sequences (default 2500).

    Example
    -------
    >>> a = torch.rand((8, 120, 512))
    >>> enc = PositionalEncoding(input_size=a.shape[-1])
    >>> b = enc(a)
    >>> b.shape
    torch.Size([1, 120, 512])
    """

    def __init__(self, input_size, max_len=2500):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(self.max_len, input_size, requires_grad=False)
        positions = torch.arange(0, self.max_len).unsqueeze(1).float()
        denominator = torch.exp(
            torch.arange(0, input_size, 2).float()
            * -(math.log(10000.0) / input_size)
        )

        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments
        ---------
        x : tensor
            Input feature shape (batch, time, fea)
        """
        return self.pe[:, : x.size(1)].clone().detach()


class LayerNorm(nn.Module):
    """Applies layer normalization to the input tensor.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input.
    eps : float
        This value is added to std deviation estimation to improve the numerical
        stability.
    elementwise_affine : bool
        If True, this module has learnable per-element affine parameters
        initialized to ones (for weights) and zeros (for biases).

    Example
    -------
    >>> input = torch.randn(100, 101, 128)
    >>> norm = LayerNorm(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 101, 128])
    """

    def __init__(
        self,
        input_size=None,
        input_shape=None,
        eps=1e-05,
        elementwise_affine=True,
    ):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if input_shape is not None:
            input_size = input_shape[2:]

        self.norm = torch.nn.LayerNorm(
            input_size,
            eps=self.eps,
            elementwise_affine=self.elementwise_affine,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channels)
            input to normalize. 3d or 4d tensors are expected.
        """
        return self.norm(x)


class MultiheadAttention(nn.Module):
    """ The class is a wrapper of MultiHead Attention for torch.nn.MultiHeadAttention.

    Reference: https://pytorch.org/docs/stable/nn.html

    Arguments
    ----------
    num_heads : int
        parallel attention heads.
    dropout : float
        a Dropout layer on attn_output_weights (default: 0.0).
    bias : bool
        add bias as module parameter (default: True).
    add_bias_kv : bool
        add bias to the key and value sequences at dim=0.
    add_zero_attn : bool
        add a new batch of zeros to the key and value sequences at dim=1.
    kdim : int
        total number of features in key (default: None).
    vdim : int
        total number of features in value (default: None).

    Example
    -------
    >>> inputs = torch.rand([8, 60, 512])
    >>> net = MultiheadAttention(nhead=8, d_model=inputs.shape[-1])
    >>> outputs, attn = net(inputs, inputs, inputs)
    >>> outputs.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        nhead,
        d_model,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
    ):
        super().__init__()

        self.att = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
        )

    def forward(
        self,
        query,
        key,
        value,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn_weights: Optional[torch.Tensor] = True,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ----------
        query : torch.Tensor
            (B, L, E) where L is the target sequence length,
            B is the batch size, E is the embedding dimension.
        key : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        value : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        key_padding_mask : torch.Tensor, optional
            (B, S) where B is the batch size, S is the source sequence
            length. If a ByteTensor is provided, the non-zero positions will
            be ignored while the position with the zero positions will be
            unchanged. If a BoolTensor is provided, the positions with the
            value of True will be ignored while the position with the value
            of False will be unchanged.
        attn_mask : torch.Tensor, optional
            2D mask (L, S) where L is the target sequence length, S is
            the source sequence length.
            3D mask (N*num_heads, L, S) where N is the batch
            size, L is the target sequence length, S is the source sequence
            length. attn_mask ensure that position i is allowed to attend the
            unmasked positions. If a ByteTensor is provided, the non-zero
            positions are not allowed to attend while the zero positions will
            be unchanged. If a BoolTensor is provided, positions with True is
            not allowed to attend while False values will be unchanged. If a
            FloatTensor is provided, it will be added to the attention weight.
        pos_embs: torch.Tensor, optional
            Positional embeddings added to the attention map of shape (L, S, E) or (L, S, 1).

        Outputs
        -------
        attn_output : torch.Tensor
            (B, L, E) where L is the target sequence length, B is the
            batch size, E is the embedding dimension.
        attn_output_weights : torch.Tensor
            (B, L, S) where B is the batch size, L is the target
            sequence length, S is the source sequence length.
        """
        # give tensors of shape (time, batch, fea)
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        # this will be legit because of https://github.com/pytorch/pytorch/blob/5288d05cfdda85c46c4df84617fa7f37c21b10b3/torch/nn/functional.py#L4946
        # we can inject relative learnable pos embeddings directly in MHA via the attn_mask
        if pos_embs is not None:
            if attn_mask is not None:
                attn_mask += pos_embs
            else:
                attn_mask = pos_embs

        output = self.att(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn_weights,
        )

        if return_attn_weights:
            output, attention_weights = output
            # reshape the output back to (batch, time, fea)
            output = output.permute(1, 0, 2)
            return output, attention_weights
        else:
            output = output.permute(1, 0, 2)
            return output


class PositionalwiseFeedForward(nn.Module):
    """The class implements the positional-wise feed forward module in
    “Attention Is All You Need”.

    Arguments
    ----------
    d_ffn: int
        Hidden layer size.
    input_shape : tuple, optional
        Expected shape of the input. Alternatively use ``input_size``.
    input_size : int, optional
        Expected size of the input. Alternatively use ``input_shape``.
    dropout: float, optional
        Dropout rate.
    activation: torch.nn.Module, optional
        activation functions to be applied (Recommendation: ReLU, GELU).

    Example
    -------
    >>> inputs = torch.rand([8, 60, 512])
    >>> net = PositionalwiseFeedForward(256, input_size=inputs.shape[-1])
    >>> outputs = net(inputs)
    >>> outputs.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_ffn,
        input_shape=None,
        input_size=None,
        dropout=0.0,
        activation=nn.ReLU,
    ):
        super().__init__()

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]

        self.ffn = nn.Sequential(
            nn.Linear(input_size, d_ffn),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, input_size),
        )

    def forward(self, x):
        # give a tensor of shap (time, batch, fea)
        x = x.permute(1, 0, 2)
        x = self.ffn(x)

        # reshape the output back to (batch, time, fea)
        x = x.permute(1, 0, 2)

        return x


# utility classes for TCN
class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)


class GAP1d(nn.Module):
    # Global Adaptive Pooling + Flatten
    def __init__(self, output_size=1):
        super(GAP1d, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = Flatten()
    def forward(self, x):
        return self.flatten(self.gap(x))


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()