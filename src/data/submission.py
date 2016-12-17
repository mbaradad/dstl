from dataset import Dataset
from classifiers.stubclassifier import StubClassifier

def generate_submission(classifier, subset=-1):
  d = Dataset(train=False)

  classifier = StubClassifier()
  image_list = d
  for im in d:
    masks = classifier.predict(d)

  # TODO: Maybe further reclassify polygons??
  # For example, classify between with a Decision tree on the area of the vehicle
  return None