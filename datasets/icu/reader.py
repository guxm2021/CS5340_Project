import pandas as pd
import numpy as np
import os

aggFuncs = {
  'Albumin': np.mean,
  'ALP': np.mean,
  'ALT': np.mean,
  'AST': np.mean,
  'Bilirubin': np.mean,
  'BUN': np.mean,
  'Cholesterol': np.mean,
  'Creatinine': np.mean,
  'DiasABP': np.mean,
  'FiO2': np.mean,
  'GCS': max,
  'Glucose': np.mean,
  'HCO3': np.mean,
  'HCT': np.mean,
  'HR': np.mean,
  'K': np.mean,
  'Lactate': np.mean,
  'Mg': np.mean,
  'MAP': np.mean,
  'MechVent': max,
  'Na': np.mean,
  'NIDiasABP': np.mean,
  'NIMAP': np.mean,
  'NISysABP': np.mean,
  'PaCO2': np.mean,
  'PaO2': np.mean,
  'pH': np.mean,
  'Platelets': np.mean,
  'RespRate': max,
  'SaO2': np.mean,
  'SysABP': np.mean,
  'Temp': np.mean,
  'TroponinI': np.mean,
  'TroponinT': np.mean,
  'Urine': sum,
  'WBC': np.mean,
  'Weight': np.mean
}

def timeToMins(t):
  p = t.split(':')
  return int(p[0]) * 60 + int(p[1])

def readPatientData(path):
  data = pd.read_csv(path)
  desc = data[data['Time']=='00:00']
  descriptors = {
    'RecordID': int(desc[desc['Parameter']=='RecordID'].values[0,2]),
    'Age': desc[desc['Parameter']=='Age'].values[0,2],
    'Gender': desc[desc['Parameter']=='Gender'].values[0,2],
    'Height': desc[desc['Parameter']=='Height'].values[0,2],
    'ICUType': desc[desc['Parameter']=='ICUType'].values[0,2],
    'Weight': desc[desc['Parameter']=='ICUType'].values[0,2],
  }
  timeseries = data.pivot_table(index='Time',columns='Parameter', values='Value', aggfunc=max)
  return  descriptors, timeseries

def initialiseEmptyDataDict(columns, maxTime, interval):
  data = {}
  numBuckets = int(np.ceil(maxTime / interval))
  for c in columns:
    data[c] = []
    for i in range(numBuckets):
      data[c].append([])
  return data

def bucketTimeseries(columns, ts, interval):
  tscolumns = ts.columns
  data = initialiseEmptyDataDict(columns, 48*60, interval)
  for idx in range(ts.shape[0]):
    bucket = int(timeToMins(ts.iloc[idx].name) // interval)
    for c in tscolumns:
      val = ts.iloc[idx][c]
      if c not in data:
        continue
      if not pd.isna(val):
        data[c][bucket].append(val)
  aggData = {}
  for c in columns:
    aggData[c] = list(map(lambda l: np.nan if len(l) == 0 else aggFuncs[c](l), data[c]))
  return pd.DataFrame(data=aggData)

def loadPatientDatasets(folder):
  datasets = {}
  columns = aggFuncs.keys()
  for f in os.listdir(folder):
    static, ts = readPatientData(os.path.join(folder, f))
    datasets[int(static['RecordID'])] = [static, bucketTimeseries(columns, ts, 60)]
  return datasets