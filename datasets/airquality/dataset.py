import pandas as pd
import numpy as np
import os
import datetime

columnIndexes = {
  'PM2.5': 0,
  'PM10': 1,
  'SO2': 2,
  'NO2': 3,
  'O3': 4,
  'TEMP': 5,
  'PRES': 6,
  'DEWP': 7,
  'WSPM': 8
}

def transformRawData(data, columnIndexes):
  '''
  Returns
  x - C x T numpy array with the data, where C is the number of columns,
      T number of readings
  m - C x T numpy array with 1 indicating the data in x is present,
      0 indicating that it is missing
  deltaT - C x T numpy array indicating the time since the last reading of the particular data point
  '''
  x = np.zeros((len(columnIndexes), data.shape[0]))
  masking = np.zeros((len(columnIndexes), data.shape[0]))
  deltaT = np.zeros((len(columnIndexes), data.shape[0]))
  tscolumns = data.columns
  timestamps = np.zeros(data.shape[0])
  for idx, values in data.iterrows():
    time = datetime.datetime(values['year'], values['month'], values['day'], values['hour'])
    timestamps[idx] = time.timestamp()
    for c in tscolumns:
      if c not in columnIndexes:
        continue
      colIdx = columnIndexes[c]
      val = values[c]
      if not pd.isna(val):
        x[colIdx, idx] = val
        masking[colIdx, idx] = 1
      if idx != 0:
        deltaT[colIdx, idx] = timestamps[idx] - timestamps[idx-1]
        if masking[colIdx, idx - 1] == 0:
          deltaT[colIdx, idx] += deltaT[colIdx, idx-1]
  return x, masking, deltaT

def loadDataset(path):
  data = pd.read_csv(path)
  return transformRawData(data)