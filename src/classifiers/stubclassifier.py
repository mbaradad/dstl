import numpy


class StubClassifier():
  def __init__(self, dataset):
    self.dataset = dataset

  #for train purposes, return always the first mask of the datset
  def predict(self, image):
    return self.dataset.generate_one(0)[1]
