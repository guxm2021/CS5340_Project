# adapted From https://github.com/locuslab/TCN and https://github.com/timeseriesAI/tsai

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from model.base import Chomp1d, GAP1d, Flatten

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

"""
c_in: number of features
c_out: number of target classes
layers set to #layers*[#hidden_units]
"""
class TCN(nn.Module):
    def __init__(self, c_in, c_out, layers=8*[25], kernel_size=7, conv_dropout=0., fc_dropout=0.):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(c_in, layers, kernel_size=kernel_size, dropout=conv_dropout)
        self.gap = GAP1d()
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else None
        self.linear = nn.Linear(layers[-1],c_out)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        #  x should be a 3D array (samples, features, time-sequence steps)
        x = self.tcn(x)
        print(x.shape)
        x = self.gap(x)
        print(x.shape)
        if self.dropout is not None: x = self.dropout(x)
        return self.linear(x)


class TCNmodel(nn.Module):
    def __init__(self, opt):
        super(TCNmodel, self).__init__()
        # define the hyper-parameters for model architecture
        self.cat_mask = opt.cat_mask
        self.cat_tp = opt.cat_tp
        if self.cat_mask:
            self.input_size = opt.input_size * 2
        else:
            self.input_size = opt.input_size
        if self.cat_tp:
            self.input_size = self.input_size + 1
        self.hidden_size = opt.hidden_size
        self.num_layers = opt.num_layers
        self.output_size = opt.output_size
        self.num_channels = self.num_layers * [self.hidden_size]
        self.kernel_size = opt.kernel_size
        # define the model architecture
        self.tcn = TemporalConvNet(num_inputs=self.input_size, num_channels=self.num_channels, kernel_size=self.kernel_size, dropout=0.0)
        self.gap = GAP1d()
        self.cls = nn.Sequential(
            # nn.Linear(self.hidden_size, self.hidden_size),
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
        # forward tcn
        x = x.transpose(1, 2).contiguous()
        x = self.tcn(x)
        # global pooling
        x = self.gap(x)                   # (B, T, F) -> (B, F)
        # x = torch.mean(x, dim=1) 
        # forward classifier
        y = self.cls(x)  # prob
        return y


class probTCN(nn.Module):
    def __init__(self, opt):
        super(probTCN, self).__init__()
        # define the hyper-parameters for model architecture
        self.cat_mask = opt.cat_mask
        self.cat_tp = opt.cat_tp
        if self.cat_mask:
            self.input_size = opt.input_size * 2
        else:
            self.input_size = opt.input_size
        if self.cat_tp:
            self.input_size = self.input_size + 1
        self.hidden_size = opt.hidden_size
        self.num_layers = opt.num_layers
        self.n_sghmc = opt.n_sghmc
        self.output_size = opt.output_size
        # define the model architecture
        self.tcn = TemporalConvNet(num_inputs=self.input_size, num_channels=self.num_channels, kernel_size=self.kernel_size, dropout=0.0)
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
        # forward tcn
        x = x.transpose(1, 2).contiguous()
        x = self.tcn(x)
        # global pooling
        # x = torch.mean(x, dim=1)  # (B, T, F) -> (B, F)
        x = self.gap(x)                   # (B, T, F) -> (B, F)
        # forward classifier
        y = []
        sp_size = (batch - 1) // len(self.cls_samples) + 1
        for _x, _cls in zip(torch.split(x, sp_size, dim=0), self.cls_samples):
            y.append(_cls(_x))
        y = torch.cat(y, dim=0)
        return y


if __name__ == "__main__":
    x = torch.randn(10, 41, 100)
    model = TCN(c_in=41, c_out=1)
    num_param = 0
    for param in model.parameters():
        num_param += param.numel()
    print(f"model {model} parameters {round(num_param / 1e3, 2)} K")
    y = model(x)
    print(y.shape)