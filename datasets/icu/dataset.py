import torch
import torch.utils.data as td
import numpy as np
import pandas as pd
import os

from datasets.icu.reader import readPatientData
from datasets.icu.process_data import timeToMins, cleanTimeseriesData, bucketTimeseries, normaliseData, aggFuncs

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
  'MechVent': 6,
  'ICUType': 7,
  'Gender': 8
}

def transformTimeSeries(ts, columnIndexes, timestamps):
  x = np.zeros((len(columnIndexes), ts.shape[0]))
  masking = np.zeros((len(columnIndexes), ts.shape[0]))
  deltaT = np.zeros((len(columnIndexes), ts.shape[0]))
  tscolumns = ts.columns
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
  def __init__(self, inputsDir, outcomesPath, columnIndexes=subsetColumnIndexes):
      self.columnIndexes = columnIndexes
      self.outcomes = pd.read_csv(outcomesPath)
      self.dir = inputsDir
      self.files = os.listdir(inputsDir)
      self.files.sort()

  def __len__(self):
      return len(self.files)

  def getOutcomes(self, recId):
    return self.outcomes[self.outcomes['RecordID'] == recId]

  def getData(self, index):
    return readPatientData(os.path.join(self.dir, self.files[index]))

  def __getitem__(self, index):
    descriptors, ts = self.getData(index)
    y = self.getOutcomes(descriptors['RecordID'])['In-hospital_death'].iloc[0]
    timestamps = list(map(timeToMins , ts.index.to_list()))
    x, masking, deltaT = transformTimeSeries(ts, self.columnIndexes, timestamps)
    return descriptors, x, masking, deltaT, y

class ProcessedDataSet(Dataset):
  def __init__(self, inputsDir, outcomesPath, columnIndexes, reference_values):
    super().__init__(inputsDir, outcomesPath, columnIndexes)
    self.referenceValues = reference_values

  def imputeData(self, x, masking):
    final_x = np.zeros((x.shape[0], x.shape[1]))
    averages = np.multiply(x, masking).sum(axis=1) / np.maximum(masking.sum(axis = 1), 1) # ensure no division by 0
    lastIndex = np.full((x.shape[0],), -1)
    for c in range(x.shape[0]):
      for t in range(x.shape[1]):
        if masking[c][t] == 1:
          final_x[c][t] = x[c][t]
          lastIndex[c] = t
        elif lastIndex[c] != -1:
          final_x[c][t] = final_x[c][lastIndex[c]]
        else:
          final_x[c][t] = averages[c]
    return final_x

  def __getitem__(self, index):
    descriptors, ts = self.getData(index)
    y = self.getOutcomes(descriptors['RecordID'])['In-hospital_death'].iloc[0]

    ts = cleanTimeseriesData(ts, self.referenceValues)
    ts = normaliseData(bucketTimeseries(self.columnIndexes.keys(), ts, 60), self.referenceValues)
    timestamps = np.arange(0, 48, 1, dtype=int)
    x, masking, _ = transformTimeSeries(ts, self.columnIndexes, timestamps)
    return torch.from_numpy(self.imputeData(x, masking)), y