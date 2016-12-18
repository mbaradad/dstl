from dataset import Dataset
from classifiers.stubclassifier import StubClassifier

def generate_submission(classifier, subset=-1):

  d = Dataset(train=False, subset=subset)
  classifier = StubClassifier(d)

  for idx in range(len(d.get_image_list())):
    im = d.generate_one(idx)
    masks = classifier.predict(d)

  # TODO: Maybe further reclassify polygons??
  # For example, classify between with a Decision tree on the area of the vehicle
  return None

if __name__ == "__main__":
  classifier = StubClassifier()
  generate_submission(classifier, subset=1)

