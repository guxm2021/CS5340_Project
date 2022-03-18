import pandas as pd
import numpy as np
import os

def readData(path):
  return pd.read_csv(path)

def loadDatasets(folder):
  datasets = []
  for f in os.listdir(folder):
    datasets.append(pd.read_csv(os.path.join(folder, f)))
  return datasets
