import pandas as pd
import numpy as np

import os

def readPatientData(path):
  data = pd.read_csv(path)
  desc = data[data['Time']=='00:00'].replace(-1, np.nan)
  descriptors = {
    'RecordID': int(desc[desc['Parameter']=='RecordID'].values[0,2]),
    'Age': desc[desc['Parameter']=='Age'].values[0,2],
    'Gender': desc[desc['Parameter']=='Gender'].values[0,2],
    'Height': desc[desc['Parameter']=='Height'].values[0,2],
    'ICUType': desc[desc['Parameter']=='ICUType'].values[0,2],
    'Weight': desc[desc['Parameter']=='ICUType'].values[0,2],
  }
  timeseries = data.pivot_table(index='Time',columns='Parameter', values='Value', aggfunc=np.array)
  return descriptors, timeseries

def loadReferenceValues(path=None):
  if path == None:
    path = os.path.join(os.path.dirname(__file__), 'data/reference-values.csv')
  return pd.read_csv(path)
