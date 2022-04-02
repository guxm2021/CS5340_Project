class TorchStandardScaler:
  def fit(self, x):
    self.mean = x.mean(1, keepdim=True)
    self.std = x.std(1, unbiased=False, keepdim=True)
  def transform(self, x):
    x -= self.mean
    x /= (self.std + 1e-8)
    return x

class TorchMinMaxScaler:
  def fit(self, x):
    self.min = x.min(1, keepdim=True)[0]
    self.max = x.max(1, keepdim=True)[0]
  def transform(self, x):
    x -= self.min
    x /= (self.max-self.min + 1e-8)
    return x