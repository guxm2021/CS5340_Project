import pandas as pd
import numpy as np

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
  'Weight': np.mean,
  'ICUType': lambda x : x[-1],
  'Gender': lambda x : x[0]
}


def timeToMins(t):
  p = t.split(':')
  return int(p[0]) * 60 + int(p[1])

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
    # Max timestamp is 48:00 bucket to the last time-bucket
    bucket = min(int(timeToMins(ts.iloc[idx].name) // interval), 47)
    for c in tscolumns:
      vals = ts.iloc[idx][c]
      if c not in data:
        continue
      if type(vals) is np.ndarray:
        data[c][bucket].extend(vals)
      elif not pd.isna(vals):
        data[c][bucket].append(vals)
  aggData = {}
  for c in columns:
    aggData[c] = list(map(lambda l: np.nan if len(l) == 0 else aggFuncs[c](l), data[c]))
  return pd.DataFrame(data=aggData)

def cleanTimeseriesData(ts, referenceVals):
  for idx in range(ts.shape[0]):
    for c in ts.columns:
      if c not in referenceVals:
        continue
      maxVal = referenceVals[c][2]
      minVal = referenceVals[c][3]
      val = ts.iloc[idx][c]
      if type(val) is np.ndarray:
        ts.iloc[idx][c] = val[np.logical_and(val <= maxVal, val >= minVal)] 
      elif val < minVal or val > maxVal:
        ts.iloc[idx][c] = np.nan
  return ts

def normaliseData(ts, referenceVals):
  for idx in range(ts.shape[0]):
    for c in ts.columns:
      if c not in referenceVals:
        continue
      norm = (ts.iloc[idx][c] - referenceVals[c][0]) / referenceVals[c][1]
      ts.iloc[idx][c] = norm
  return ts