import pandas as pd
import numpy as np

aggsFuncs = {
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
  descriptors = data[data['Time']=='00:00'].pivot(index='Time',columns='Parameter', values='Value')
  timeseries = data[data['Time']!='00:00'].pivot_table(index='Time',columns='Parameter', values='Value', aggfunc=max)
  return  descriptors, timeseries

def initialiseEmptyDataDict(maxTime, interval):
  data = {}
  numBuckets = int(np.ceil(maxTime / interval))
  for c in aggsFuncs.keys():
    data[c] = []
    for i in range(numBuckets):
      data[c].append([])
  return data

def bucketTimeseries(ts, interval):
  columns = ts.columns
  data = initialiseEmptyDataDict(48*60, interval)
  for idx in range(ts.shape[0]):
    bucket = int(timeToMins(ts.iloc[idx].name) // interval)
    for c in columns:
      val = ts.iloc[idx][c]
      if not pd.isna(val):
        data[c][bucket].append(val)
  aggData = {}
  for c in aggsFuncs.keys():
    aggData[c] = list(map(lambda l: np.nan if len(l) == 0 else aggsFuncs[c](l), data[c]))
  return pd.DataFrame(data=aggData)

def readPatientOutcomes(path):
  return pd.read_csv(path)