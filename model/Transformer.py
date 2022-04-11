import torch
import torch.nn as nn
from typing import Optional
from model.base import PositionalEncoding, LayerNorm, MultiheadAttention, PositionalwiseFeedForward, GAP1d


class TransformerEncoderLayer(nn.Module):
    """This is an implementation of self-attention encoder layer.

    Arguments
    ----------
    d_ffn: int, optional
        The dimension of the feedforward network model hidden layer.
    nhead: int
        The number of heads in the multi-head attention models (default=8).
    d_model: int
        The number of expected features in the encoder/decoder inputs (default=512).
    kdim: int, optional
        Dimension of the key.
    vdim: int, optional
        Dimension of the value.
    dropout: int, optional
        The dropout value.
    activation: torch.nn.Module, optional
        The activation function for Feed-Forward Netowrk layer,
        e.g., relu or gelu or swish.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    attention_type: str, optional
        Type of attention layer used in all Transformer or Conformer layers.
        e.g. regularMHA or RelPosMHA.

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = TransformerEncoderLayer(512, 8, d_model=512)
    >>> output = net(x)
    >>> output[0].shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_ffn,
        nhead,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        causal=False,
    ):
        super().__init__()

        self.self_att = MultiheadAttention(
                nhead=nhead,
                d_model=d_model,
                dropout=dropout,
                kdim=kdim,
                vdim=vdim,
        )

        self.pos_ffn = PositionalwiseFeedForward(
            d_ffn=d_ffn,
            input_size=d_model,
            dropout=dropout,
            activation=activation,
        )

        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.norm2 = LayerNorm(d_model, eps=1e-6)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder layer.
        src_mask : torch.Tensor
            The mask for the src query for each example in the batch.
        src_key_padding_mask : torch.Tensor, optional
            The mask for the src keys for each example in the batch.
        """
        if self.normalize_before:
            src1 = self.norm1(src)
        else:
            src1 = src

        output, self_attn = self.self_att(
            src1,
            src1,
            src1,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs,
        )

        # add & norm
        src = src + self.dropout1(output)
        if not self.normalize_before:
            src = self.norm1(src)

        if self.normalize_before:
            src1 = self.norm2(src)
        else:
            src1 = src
        output = self.pos_ffn(src1)

        # add & norm
        output = src + self.dropout2(output)
        if not self.normalize_before:
            output = self.norm2(output)

        return output, self_attn


class TransformerEncoder(nn.Module):
    """This class implements the transformer encoder.

    Arguments
    ---------
    num_layers : int
        Number of transformer layers to include.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    d_model : int
        The dimension of the input embedding.
    kdim : int
        Dimension for key (Optional).
    vdim : int
        Dimension for value (Optional).
    dropout : float
        Dropout for the encoder (Optional).
    input_module: torch class
        The module to process the source input feature to expected
        feature dimension (Optional).

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = TransformerEncoder(1, 8, 512, d_model=512)
    >>> output, _ = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        input_shape=None,
        d_model=None,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        causal=False,
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    causal=causal,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ----------
        src : tensor
            The sequence to the encoder layer (required).
        src_mask : tensor
            The mask for the src sequence (optional).
        src_key_padding_mask : tensor
            The mask for the src keys per batch (optional).
        """
        output = src
        attention_lst = []
        for enc_layer in self.layers:
            output, attention = enc_layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                pos_embs=pos_embs,
            )
            attention_lst.append(attention)
        output = self.norm(output)

        return output, attention_lst


class Transformermodel(nn.Module):
    def __init__(self, opt):
        super(Transformermodel, self).__init__()
        self.cat_mask = opt.cat_mask
        self.cat_tp = opt.cat_tp
        if self.cat_mask:
            self.input_size = opt.input_size * 2
        else:
            self.input_size = opt.input_size
        if self.cat_tp:
            self.input_size = self.input_size + 1
        self.hidden_size = opt.hidden_size
        self.output_size = opt.output_size
        self.num_layers = opt.num_layers
        self.proj = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
        )
        self.encoder = TransformerEncoder(num_layers=self.num_layers, nhead=4, d_ffn=self.hidden_size, d_model=self.hidden_size)
        self.positional_encoding = PositionalEncoding(input_size=self.hidden_size, max_len=5000)
        self.gap = GAP1d()
        self.cls = nn.Sequential(
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Sigmoid(),
        )

    def forward(self, x, tp=None, mask=None):
        batch, frame, _ = x.size()
        if self.cat_mask:
            x = torch.cat([x, mask], dim=-1)
        if self.cat_tp:
            x = torch.cat([x, tp], dim=-1)
        # forward transformer
        x = self.proj(x)
        x = x + self.positional_encoding(x)
        x, attn_lst = self.encoder(x)
        # global pooling
        # x = torch.mean(x, dim=1)  # (B, T, F) -> (B, F)
        x = x.transpose(1, 2).contiguous()
        x = self.gap(x)                   # (B, T, F) -> (B, F)
        # forward classifier
        y = self.cls(x)
        return y


class probTransformer(nn.Module):
    def __init__(self, opt):
        super(probTransformer, self).__init__()
        self.cat_mask = opt.cat_mask
        self.cat_tp = opt.cat_tp
        if self.cat_mask:
            self.input_size = opt.input_size * 2
        else:
            self.input_size = opt.input_size
        if self.cat_tp:
            self.input_size = self.input_size + 1
        self.hidden_size = opt.hidden_size
        self.output_size = opt.output_size
        self.num_layers = opt.num_layers
        self.n_sghmc = opt.n_sghmc
        self.proj = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
        )
        self.encoder = TransformerEncoder(num_layers=self.num_layers, nhead=8, d_ffn=self.hidden_size, d_model=self.hidden_size)
        self.positional_encoding = PositionalEncoding(input_size=self.hidden_size, max_len=5000)
        self.gap = GAP1d()
        self.cls_samples = []
        for i in range(opt.n_sghmc):
            cls = nn.Sequential(
                #   nn.Linear(self.hidden_size, self.hidden_size),
                #   nn.ReLU(inplace=True),
                  nn.Linear(self.hidden_size, self.output_size),
                  nn.Sigmoid(),
                  )
            setattr(self, 'cls_{}'.format(i), cls)
            self.cls_samples.append(cls)


    def forward(self, x, tp=None, mask=None):
        batch, frame, _ = x.size()
        if self.cat_mask:
            x = torch.cat([x, mask], dim=-1)
        if self.cat_tp:
            x = torch.cat([x, tp], dim=-1)
        # forward transformer
        x = self.proj(x)
        x = x + self.positional_encoding(x)
        x, attn_lst = self.encoder(x)
        # global pooling
        # x = torch.mean(x, dim=1)  # (B, T, F) -> (B, F)
        x = x.transpose(1, 2).contiguous()
        x = self.gap(x)                   # (B, T, F) -> (B, F)
        # forward classifier
        y = []
        sp_size = (batch - 1) // len(self.cls_samples) + 1
        for _x, _cls in zip(torch.split(x, sp_size, dim=0), self.cls_samples):
            y.append(_cls(_x))
        y = torch.cat(y, dim=0)
        return y


# if __name__ == "__main__":
#     x = torch.randn(40, 10, 20)
#     model = Transformermodel(num_layers=1)
#     y = model(x)
#     print(y.shape)
    