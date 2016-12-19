import numpy


class StubClassifier():
  def __init__(self, dataset):
    self.dataset = dataset

  def predict(self, image):
    return self.dataset.generate_one(0)[1]
