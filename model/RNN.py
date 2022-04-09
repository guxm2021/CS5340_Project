import torch
import torch.nn as nn


class GRUmodel(nn.Module):
    def __init__(self, num_layers, input_size=20, hidden_size=1024, output_size=1):
        super(GRUmodel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False)
        self.cls = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        batch, frame, _ = x.size()
        # forward rnn
        x, hidden = self.rnn(x)
        # global pooling
        x = torch.mean(x, dim=1)  # (B, T, F) -> (B, F)
        # forward classifier
        y = self.cls(x)
        return y


class LSTMmodel(nn.Module):
    def __init__(self, num_layers, input_size=20, hidden_size=1024, output_size=1):
        super(LSTMmodel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False)
        self.cls = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        batch, frame, _ = x.size()
        # forward rnn
        x, hidden = self.rnn(x)
        # global pooling
        x = torch.mean(x, dim=1)  # (B, T, F) -> (B, F)
        # forward classifier
        y = self.cls(x)
        return y


class probGRU(nn.Module):
    def __init__(self, n_sghmc, num_layers, input_size=20, hidden_size=1024, output_size=1):
        super(probGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_sghmc = n_sghmc
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False)
        self.cls_samples = []
        for i in range(n_sghmc):
            cls = nn.Sequential(
                  nn.Linear(hidden_size, hidden_size),
                  nn.ReLU(inplace=True),
                  nn.Linear(hidden_size, output_size)
                  )
            setattr(self, 'cls_{}'.format(i), cls)
            self.cls_samples.append(cls)


    def forward(self, x):
        batch, frame, _ = x.size()
        # forward rnn
        x, hidden = self.rnn(x)
        # global pooling
        x = torch.mean(x, dim=1)  # (B, T, F) -> (B, F)
        # forward classifier
        y = []
        sp_size = (batch - 1) // len(self.cls_samples) + 1
        for _x, _cls in zip(torch.split(x, sp_size, dim=0), self.cls_samples):
            y.append(_cls(_x))
        y = torch.cat(y, dim=0)
        return y


class probLSTM(nn.Module):
    def __init__(self, n_sghmc, num_layers, input_size=20, hidden_size=1024, output_size=1):
        super(probLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_sghmc = n_sghmc
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False)
        self.cls_samples = []
        for i in range(n_sghmc):
            cls = nn.Sequential(
                  nn.Linear(hidden_size, hidden_size),
                  nn.ReLU(inplace=True),
                  nn.Linear(hidden_size, output_size)
                  )
            setattr(self, 'cls_{}'.format(i), cls)
            self.cls_samples.append(cls)


    def forward(self, x):
        batch, frame, _ = x.size()
        # forward rnn
        x, hidden = self.rnn(x)
        # global pooling
        x = torch.mean(x, dim=1)  # (B, T, F) -> (B, F)
        # forward classifier
        y = []
        sp_size = (batch - 1) // len(self.cls_samples) + 1
        for _x, _cls in zip(torch.split(x, sp_size, dim=0), self.cls_samples):
            y.append(_cls(_x))
        y = torch.cat(y, dim=0)
        return y


if __name__ == "__main__":
    x = torch.randn(40, 10, 20)
    num_layers = 2
    n_sghmc = 8
    # model = GRUmodel(num_layers)
    # model = LSTMmodel(num_layers)
    # model = probGRU(n_sghmc, num_layers)
    model = probLSTM(n_sghmc, num_layers)
    y = model(x)
    print(y.shape)
