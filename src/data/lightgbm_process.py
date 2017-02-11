from dataset import Dataset
from classifiers.resnet_keras.classifier import ResnetClassifier
import pycocotools.mask as mask
import numpy as np
from utils.utils import *
from shapely.affinity import scale
from sima.ROI import mask2poly, poly2mask
from utils.utils import normal_coordinates_to_dataset_coordinates
from shapely.ops import cascaded_union, unary_union
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.collection import GeometryCollection
from shapely.geometry.polygon import Polygon
from shapely.wkt import loads, dumps
from shapely.ops import triangulate
import datetime
from utils.dirs import *

from submission import get_header

import cv2
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import math
from repair_topology import __get_valid_wkt_str
from itertools import product
from result_generator import ResultGenerator
from pylightgbm.models import GBMClassifier
from sklearn import datasets, metrics, model_selection

def produce_lightgbm():

  classes_list = [p for p in product([0, 1], repeat=10)]
  classes = dict(zip(classes_list, range(len(classes_list))))

  #DON'T USE SIMPLIFYYYY
  #THE PROBLEMS IS POLYGONS OF 1 PIXEL/LINES
  #OR MAYBE USE TRIANGULATE
  r = ResultGenerator(train=True, store_processed_images=False)
  d = Dataset(train=True, store_processed_images=False)
  image_list = r.image_list
  X = list()

  #for i in range(len(image_list)):
  i = 0
  predicted_masks = r.generate_one(i)
  _, gt_masks = d.generate_one(i)
  X = np.zeros([predicted_masks[0].shape[0] * predicted_masks[0].shape[1]/10, 10])
  Y = np.zeros([predicted_masks[0].shape[0] * predicted_masks[0].shape[1]/10])
  for k in range(predicted_masks[0].shape[0]):
    print 'row' + str(k)
    for l in range(predicted_masks[0].shape[1]/10):
      gt_pixel = np.zeros(10)
      for c in range(10):
        X[k*predicted_masks[0].shape[1]/10 + l, c] = predicted_masks[c][k,l]
        gt_pixel[c] = (int(gt_masks[c,k,l]))
      Y[k * predicted_masks[0].shape[1]/10 + l] = classes[tuple(gt_pixel)]

  executable_path = "lightgbm"

  x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2)

  clf = GBMClassifier(exec_path=executable_path, min_data_in_leaf = 1, application='multiclass', num_class=len(classes_list), metric='multi_logloss')
  clf.fit(x_train, y_train, test_data=[(x_test, y_test)])
  y_pred = clf.predict(x_test)
  print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))


if __name__ == "__main__":
  produce_lightgbm()

