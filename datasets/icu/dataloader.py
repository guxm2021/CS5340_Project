import torch
import torch.utils.data as td
import os

from datasets.icu.dataset import ProcessedDataSet, subsetColumnIndexes
from datasets.icu.sampler import Sampler
from datasets.icu.reader import loadReferenceValues

def getProcessedDataloaders(trainDir, trainYPath, trainDataInfoPath, testDir, testYPath, testDataInfoPath, batchSize=64):
  refVals = loadReferenceValues()
  kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}

  trainData = ProcessedDataSet(trainDir, trainYPath, subsetColumnIndexes, refVals)
  trainSampler = Sampler(trainDataInfoPath, subsetColumnIndexes.keys())
  trainDataloader = td.DataLoader(dataset=trainData, sampler=trainSampler, shuffle=False, batch_size=batchSize, **kwargs)

  testData = ProcessedDataSet(testDir, testYPath, subsetColumnIndexes, refVals)
  testSampler = Sampler(testDataInfoPath, subsetColumnIndexes.keys())
  testDataloader = td.DataLoader(dataset=testData, sampler=testSampler, shuffle=False, batch_size=batchSize, **kwargs)
  return trainDataloader, testDataloader

if __name__ == '__main__':  
  base = os.path.dirname(__file__)
  args = {
    # Training data paths
    'trainDir': os.path.join(base, 'data/training'),
    'trainYPath': os.path.join(base,'data/Outcomes-training.txt'), 
    'trainDataInfoPath': os.path.join(base, 'data/sample-training.csv'), 
    
    # Testing data paths
    'testDir': os.path.join(base,'data/testing'),
    'testYPath': os.path.join(base, 'data/Outcomes-test.txt'), 'testDataInfoPath':  os.path.join(base, 'data/sample-test.csv'),

    'batchSize': 1024
  }
  trainDataloader, testDataloader = getProcessedDataloaders(**args)
  # for idx, (_, target) in enumerate(trainDataloader):
  #   print('training batch', idx + 1)
  #   print(target)

  # for idx, (_, target) in enumerate(testDataloader):
  #   print('testing batch', idx + 1)
  #   print(target)
    