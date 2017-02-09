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

def process_submission():
  #DON'T USE SIMPLIFYYYY
  #THE PROBLEMS IS POLYGONS OF 1 PIXEL/LINES
  #OR MAYBE USE TRIANGULATE
  submission = '/home/manel/Documents/dstl/output/submissions/submission_2017-02-09_01:26:52.273168.csv'
  df = pd.read_csv(submission)
  f = open(submission.replace(".csv", "modified.csv") , 'w')
  f.write(get_header() + '\n')  # python will convert \n to os.linesep
  i = 0
  d = Dataset(False)
  for im in d.images_to_submit:
    for c in range(1,11):
      row = df.loc[df['ImageId'] == im].loc[df['ClassType'] == c]
      f.write(",".join(
        [str(row['ImageId'].iloc[0]), str(row['ClassType'].iloc[0]), '"' + str(row['MultipolygonWKT'].iloc[0]) + '"']) + '\n')
    continue
  return
  for row in df.iterrows():
    print i
    if str(row[1]['ImageId']) in d.images_to_submit:
      f.write(",".join(
        [str(row[1]['ImageId']), str(row[1]['ClassType']), '"' + str(row[1]['MultipolygonWKT']) + '"']) + '\n')
    # unary_union
    # DON't use simplify!!!!
    polygons = loads(row[1]['MultipolygonWKT'])
    i = i + 1
    continue
    if type(polygons) == Polygon:
      p = polygons
      polygons = list()
      polygons.append(p)
    filtered_polygons = list()
    area_discarded = 0
    total_area = 1e-32
    for k in polygons:
        for p in triangulate(k, 5e-6):
          if p.area > 5e-12:
            filtered_polygons.append(p)
          else:
            area_discarded = area_discarded + p.area
        total_area = total_area + k.area
    polygons = cascaded_union(filtered_polygons)
    if type(polygons) == Polygon:
      p = polygons
      polygons = list()
      polygons.append(p)
    print 'Area discarded: ' + str(area_discarded/total_area)
    row[1]['MultipolygonWKT'] = __get_valid_wkt_str(MultiPolygon(polygons), 6)
    #row[1]['MultipolygonWKT'] = __get_valid_wkt_str(loads(row[1]['MultipolygonWKT']))
    f.write(",".join(
      [str(row[1]['ImageId']), str(row[1]['ClassType']), '"' + str(row[1]['MultipolygonWKT']) + '"']) + '\n')


if __name__ == "__main__":

  process_submission()

