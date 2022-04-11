import torch
import torch.nn as nn
from model.base import GAP1d

class ODERNNmodel(nn.Module):
    def __init__(self, opt):
        super(ODERNNmodel, self).__init__()
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
        self.n_step = opt.n_step

        # define the model architecture
        self.layers = []
        for layer in range(self.num_layers):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            if layer < self.num_layers - 1:
                self.layers.append(nn.ReLU(inplace=True))
        self.ode_func = nn.Sequential(*self.layers)
        self.rnn_cell = nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)
        self.gap = GAP1d()
        self.cls = nn.Sequential(
            # nn.Linear(self.hidden_size, self.hidden_size),
            # nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Sigmoid(),
        )
    
    def forward(self, x, tp=None, mask=None):
        # x: (B, T, 41)  tp: (B, T, 1)  mask: (B, T, 41)
        batch, frame, _ = x.size()
        hidden_states = []
        hi = torch.zeros(batch, self.hidden_size).float().to(x.device)  # initialize 
        for i in range(frame):
            xi = x[:, i, :]
            if i == 0:
                hi_prime = hi
            else:
                ti_last = tp[:, i-1]    # (B, 1)
                ti = tp[:, i]         # (B, 1)
                hi_prime = self.forward_ode(hi_last, ti_last, ti)
            
            hi = self.rnn_cell(xi, hi_prime)
            hi_last = hi
            hidden_states.append(hi.unsqueeze(dim=2))
        x = torch.cat(hidden_states, dim=2)  # (B, T, F)
        # global pooling
        x = self.gap(x)                   # (B, T, F) -> (B, F)
        # x = torch.mean(x, dim=1) 
        # forward classifier
        y = self.cls(x)  # prob
        return y
    
    def forward_ode(self, hi_last, ti_last, ti):
        # hi_last: last hidden_state
        # ti_last: last time stamp
        # ti: current time stamp
        hi = hi_last
        for _ in range(self.n_step):
            grad = self.ode_func(hi)
            hi = hi + grad * (ti - ti_last) / self.n_step
        return hi


class probODERNN(nn.Module):
    def __init__(self, opt):
        super(probODERNN, self).__init__()
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
        self.n_step = opt.n_step

        # define the model architecture
        self.layers = []
        for layer in range(self.num_layers):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            if layer < self.num_layers - 1:
                self.layers.append(nn.ReLU(inplace=True))
        self.ode_func = nn.Sequential(*self.layers)
        self.rnn_cell = nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)
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
        # x: (B, T, 41)  tp: (B, T, 1)  mask: (B, T, 41)
        batch, frame, _ = x.size()
        hidden_states = []
        hi = torch.zeros(batch, self.hidden_size).float().to(x.device)  # initialize 
        for i in range(frame):
            xi = x[:, i, :]
            if i == 0:
                hi_prime = hi
            else:
                ti_last = tp[:, i-1]    # (B, 1)
                ti = tp[:, i]         # (B, 1)
                hi_prime = self.forward_ode(hi_last, ti_last, ti)
            
            hi = self.rnn_cell(xi, hi_prime)
            hi_last = hi
            hidden_states.append(hi.unsqueeze(dim=2))
        x = torch.cat(hidden_states, dim=2)  # (B, T, F)
        # global pooling
        x = self.gap(x)                   # (B, T, F) -> (B, F)
        # x = torch.mean(x, dim=1) 
        # forward classifier
        y = []
        sp_size = (batch - 1) // len(self.cls_samples) + 1
        for _x, _cls in zip(torch.split(x, sp_size, dim=0), self.cls_samples):
            y.append(_cls(_x))
        y = torch.cat(y, dim=0)
        return y
    
    def forward_ode(self, hi_last, ti_last, ti):
        # hi_last: last hidden_state
        # ti_last: last time stamp
        # ti: current time stamp
        hi = hi_last
        for _ in range(self.n_step):
            grad = self.ode_func(hi)
            hi = hi + grad * (ti - ti_last) / self.n_step
        return hi
