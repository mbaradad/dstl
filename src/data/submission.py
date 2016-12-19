from dataset import Dataset
from classifiers.stubclassifier import StubClassifier
import pycocotools.mask as mask
import numpy as np
from utils.utils import *
from sima.ROI import mask2poly, poly2mask
from utils.utils import normal_coordinates_to_dataset_coordinates
from shapely.ops import cascaded_union
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon
import datetime
from utils.dirs import *

def generate_submission(classifier, subset=-1):

  d = Dataset(train=False, subset=subset)
  now = datetime.datetime.now()
  f = open(SUBMISSION + '/submission_' + str(now).replace(" ", "_"), 'w')
  f.write(get_header() + '\n')  # python will convert \n to os.linesep

  for idx in range(len(d.get_image_list())):
    im, _ = d.generate_one(idx)

    # TODO: Maybe further reclassify polygons after main mask classifier??
    # For example, classify between small/large vehicles by area using Decision trees,
    # based on training samples, directly from polygons
    masks = classifier.predict(im)

    height = masks.shape[1]
    width = masks.shape[2]
    [x_max, y_min] = d.get_grid_size(idx)
    for i in range(len(masks)):
      polygons = mask2poly(masks[i])
      polygons = cascaded_union(polygons)
      normalized_polygons = list()
      for polygon in polygons:
        transformed_coordinates = [normal_coordinates_to_dataset_coordinates(x, y, height, width, x_max, y_min)
                                   for (x, y, _) in np.array(polygon.exterior)]
        normalized_polygons.append(Polygon(transformed_coordinates))
      multiPolygon = MultiPolygon(normalized_polygons)
      f.write(d.get_image_list()[idx] + ',' + str(i + 1) + ',' + multiPolygon.wkt + '\n')

  f.close()

def get_header():
  return "ImageId,ClassType,MultipolygonWKT"

if __name__ == "__main__":
  d = Dataset(train=True, subset=1)
  classifier = StubClassifier(d)
  generate_submission(classifier, subset=1)

