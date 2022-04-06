from sympy import subsets
import torch.utils.data as td
import pandas as pd
import numpy as np

class Sampler(td.Sampler):
  def __init__(self, datainfo_file, filter_columns=[]):
    self.data = pd.read_csv(datainfo_file)
    data = self.data.values
    data = data[data[:, 0].argsort()]
    # (index, recordID, number of timestamps)
    files = np.array([(i, data[i, 0], data[i, 1]) for i in range(len(data))])
    self.files = self._filtered_files(files, self.data, filter_columns)
  
  def _filtered_files(self, files, data, columns):
    if len(columns) == 0:
      return files
    res = []
    for t in files:
      counts = data[data['RecordID']==t[1]]
      skip = False
      for col in columns:
        if counts[col].any() == 0:
          skip = True
          break
      if not skip:
        res.append(t)
    return np.array(res)
      
  def __iter__(self):
    files = self.files
    indices = files[:,0]
    np.random.shuffle(indices)
    yield from iter(indices)

  def __len__(self):
    return len(self.files)

class SizeBatchedSampler(Sampler):
  def __init__(self, datainfo_file, filter_columns, size):
    super().__init__(datainfo_file, filter_columns)
    self.size = size
    
  def __iter__(self):
    files = self.files
    indices = files[(-1 * files[:,2]).argsort()][:,0]
    batches = np.array_split(indices, len(self))
    for b in batches:
      np.random.shuffle(b)
      yield b
    
  def __len__(self):
    return int(np.ceil(len(self.files) / self.size))
