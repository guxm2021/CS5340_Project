import torch.utils.data as td
import pandas as pd
import numpy as np

class Sampler(td.Sampler):
  def __init__(self, datainfo_file, size):
    self.size = size
    data = pd.read_csv(datainfo_file).values
    data = data[data[:, 0].argsort()]
    # (index, recordID, number of timestamps)
    self.files = np.array([(i, data[i, 0], data[i, 1]) for i in range(len(data))])

  def __iter__(self):
    indices = self.files[(-1 * self.files[:,2]).argsort()][:,0]
    batches = np.array_split(indices, len(self))
    for b in batches:
      np.random.shuffle(b)
      yield b

  def __len__(self):
    return int(np.ceil(len(self.files) / self.size))
    
