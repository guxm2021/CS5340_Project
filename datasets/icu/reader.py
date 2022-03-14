import pandas as pd

def readPatientData(path):
  data = pd.read_csv(path)
  descriptors = data[data['Time']=='00:00'].pivot(index='Time',columns='Parameter', values='Value')
  timeseries = data[data['Time']!='00:00'].pivot_table(index='Time',columns='Parameter', values='Value', aggfunc=max)
  timeseries['Mins'] = timeseries.index.to_series().str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
  return  descriptors, timeseries


def readPatientOutcomes(path):
  return pd.read_csv(path)