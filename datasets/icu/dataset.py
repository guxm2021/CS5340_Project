import torch.utils.data as td
import numpy as np
import pandas as pd
import os

from reader import readPatientData
from process_data import timeToMins, aggFuncs

columnIndexes = {
  'Albumin': 0,
  'ALP': 1,
  'ALT': 2,
  'AST': 3,
  'Bilirubin': 4,
  'BUN': 5,
  'Cholesterol': 6,
  'Creatinine': 7,
  'DiasABP': 8,
  'FiO2': 9,
  'GCS': 10,
  'Glucose': 11,
  'HCO3': 12,
  'HCT': 13,
  'HR': 14,
  'K': 15,
  'Lactate': 16,
  'Mg': 17,
  'MAP': 18,
  'MechVent': 19,
  'Na': 20,
  'NIDiasABP': 21,
  'NIMAP': 22,
  'NISysABP': 23,
  'PaCO2': 24,
  'PaO2': 25,
  'pH': 26,
  'Platelets': 27,
  'RespRate': 28,
  'SaO2': 29,
  'SysABP': 30,
  'Temp': 31,
  'TroponinI': 32,
  'TroponinT': 33,
  'Urine': 34,
  'WBC': 35,
  'Weight': 36 
}

#  The according to a study, the columns that matter are - 
#  BUN, GCS, HCO3, HCT, Na, Urine, MechVent, ICUType (non-timeseries), Gender (non-timeseries)
subsetColumnIndexes = {
  'BUN': 0,
  'GCS': 1,
  'HCO3': 2,
  'HCT': 3,
  'Na': 4,
  'Urine': 5,
  'MechVent': 6
}

def transformTimeSeries(ts, columnIndexes):
  x = np.zeros((len(columnIndexes), ts.shape[0]))
  masking = np.zeros((len(columnIndexes), ts.shape[0]))
  deltaT = np.zeros((len(columnIndexes), ts.shape[0]))
  tscolumns = ts.columns
  timestamps = list(map(timeToMins , ts.index.to_list()))
  for t, values in ts.iterrows():
    tIdx = ts.index.get_loc(t)
    for c in tscolumns:
      if c not in columnIndexes:
        continue
      colIdx = columnIndexes[c]
      val = values[c]
      if type(val) is np.ndarray:
        val = aggFuncs[c](val)
      if not pd.isna(val):
        x[colIdx, tIdx] = val
        masking[colIdx, tIdx] = 1
      if tIdx != 0:
        deltaT[colIdx, tIdx] = timestamps[tIdx] - timestamps[tIdx-1]
        if masking[colIdx, tIdx - 1] == 0:
          deltaT[colIdx, tIdx] += deltaT[colIdx, tIdx-1]
  return x, masking, deltaT

class Dataset(td.Dataset):
    def __init__(self, inputs_dir, outcomes_file, columnIndexes=subsetColumnIndexes):
        self.columnIndexes = columnIndexes
        self.outcomes = pd.read_csv(outcomes_file)
        self.dir = inputs_dir
        self.files = os.listdir(inputs_dir)
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
      descriptors, ts = readPatientData(os.path.join(self.dir, self.files[index]))
      recId = descriptors['RecordID']
      outcome = self.outcomes[self.outcomes['RecordID'] == recId]
      y = outcome['In-hospital_death'].iloc[0]
      x, masking, deltaT = transformTimeSeries(ts, self.columnIndexes)
      return descriptors, x, masking, deltaT, y
