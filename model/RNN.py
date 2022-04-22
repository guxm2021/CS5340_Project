import torch
import torch.nn as nn
from model.base import GAP1d


class GRUmodel(nn.Module):
    def __init__(self, opt):
        super(GRUmodel, self).__init__()
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
        
        # define the model architecture
        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, 
                          num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.gap = GAP1d()
        self.cls = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, x, tp=None, mask=None):
        batch, frame, _ = x.size()
        if self.cat_mask:
            x = torch.cat([x, mask], dim=-1)
        if self.cat_tp:
            x = torch.cat([x, tp], dim=-1)
        # forward rnn
        x, hidden = self.rnn(x)
        # global pooling
        # x = torch.mean(x, dim=1)  # (B, T, F) -> (B, F)
        x = x.transpose(1, 2).contiguous()
        x = self.gap(x)                   # (B, T, F) -> (B, F)
        # forward classifier
        y = self.cls(x)  # prob
        return y


class LSTMmodel(nn.Module):
    def __init__(self, opt):
        super(LSTMmodel, self).__init__()
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
        
        # define the model architecture
        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, 
                           num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.gap = GAP1d()
        self.cls = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, x, tp=None, mask=None):
        batch, frame, _ = x.size()
        if self.cat_mask:
            x = torch.cat([x, mask], dim=-1)
        if self.cat_tp:
            x = torch.cat([x, tp], dim=-1)
        # forward rnn
        x, hidden = self.rnn(x)
        # global pooling
        # x = torch.mean(x, dim=1)  # (B, T, F) -> (B, F)
        x = x.transpose(1, 2).contiguous()
        x = self.gap(x)                   # (B, T, F) -> (B, F)
        # forward classifier
        y = self.cls(x)
        return y


class probLSTM(nn.Module):
    def __init__(self, opt):
        super(probLSTM, self).__init__()
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
        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, 
                           num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.gap = GAP1d()
        self.cls_samples = []
        for i in range(opt.n_sghmc):
            cls = nn.Linear(self.hidden_size * 2, self.output_size)
            setattr(self, 'cls_{}'.format(i), cls)
            self.cls_samples.append(cls)


    def forward(self, x, tp=None, mask=None):
        batch, frame, _ = x.size()
        if self.cat_mask:
            x = torch.cat([x, mask], dim=-1)
        if self.cat_tp:
            x = torch.cat([x, tp], dim=-1)
        # forward rnn
        x, hidden = self.rnn(x)
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

class probGRU(nn.Module):
    def __init__(self, opt):
        super(probGRU, self).__init__()
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
        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, 
                          num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.gap = GAP1d()
        self.cls_samples = []
        for i in range(opt.n_sghmc):
            cls = nn.Linear(self.hidden_size * 2, self.output_size)
            setattr(self, 'cls_{}'.format(i), cls)
            self.cls_samples.append(cls)


    def forward(self, x, tp=None, mask=None):
        batch, frame, _ = x.size()
        if self.cat_mask:
            x = torch.cat([x, mask], dim=-1)
        if self.cat_tp:
            x = torch.cat([x, tp], dim=-1)
        # forward rnn
        x, hidden = self.rnn(x)
        # global pooling
        # x = torch.mean(x, dim=1)  # (B, T, F) -> (B, F)
        x = x.transpose(1, 2).contiguous()
        x = self.gap(x)                   # (B, T, F) -> (B, F)
        # forward classifier
        y = []
        for cls in self.cls_samples:
            y.append(cls(x).unsqueeze(dim=1))

        y = torch.cat(y, dim=1)           # (B, N, 2)
        y = torch.mean(y, dim=1)          # (B, 2)
        return y

# if __name__ == "__main__":
#     x = torch.randn(40, 10, 20)
#     num_layers = 2
#     n_sghmc = 8
#     # model = GRUmodel(num_layers)
#     # model = LSTMmodel(num_layers)
#     # model = probGRU(n_sghmc, num_layers)
#     model = probLSTM(n_sghmc, num_layers)
#     y = model(x)
#     print(y.shape)
