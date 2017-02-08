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
import shapely
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import math
from repair_topology import __get_valid_wkt_str
import numpy as np

def process_submission():
  kernel_closings = [5, 5, 10, 20, 10, 10, 10, 10, 10, 10]
  kernel_openings = [10, 6, 7, 7, 10, 10, 2, 2, 2, 2]

  #DON'T USE SIMPLIFYYYY
  #THE PROBLEMS IS POLYGONS OF 1 PIXEL/LINES
  #OR MAYBE USE TRIANGULATE
  submission = '/home/manel/Documents/dstl/output/submissions/submission_2017-02-07_02:24:07.905275.csv'
  df = pd.read_csv(submission)
  f = open(submission.replace(".csv", "from_np.csv") , 'w')
  np_dir = submission.replace(".csv", "/")
  f.write(get_header() + '\n')  # python will convert \n to os.linesep
  i = 0
  d = Dataset(train=False)

  for row in df.iterrows():
    dataset_image_id = d.get_image_list()[i/10]
    [x_max, y_min] = d.get_grid_size(i/10)
    print i
    i = i + 1
    #unary_union
    #DON't use simplify!!!!
    image_id = row[1]['ImageId']
    if image_id != dataset_image_id:
      raise Exception('Dataset id is not the same as previous submission')
    class_id = row[1]['ClassType']
    actual_mask_path = np_dir + 'mask_' + str(image_id) + '_' + str(class_id-1) + '.h5.npy'
    actual_mask = np.load(actual_mask_path)
    actual_mask = np.asarray(actual_mask, dtype='uint8')

    # minimum area to work
    if kernel_closings[class_id-1] != 0:
      kernel_close = np.ones((kernel_closings[class_id-1], kernel_closings[class_id-1]), np.uint8)
      actual_mask = cv2.morphologyEx(actual_mask, cv2.MORPH_CLOSE, kernel_close)
      kernel_open = np.ones((kernel_openings[class_id-1], kernel_openings[class_id-1]), np.uint8)
      actual_mask = cv2.morphologyEx(actual_mask, cv2.MORPH_OPEN, kernel_open)

    polygons = mask_to_polygons(actual_mask, epsilon=5, min_area=5)

    height = actual_mask.shape[0]
    width = actual_mask.shape[1]

    multiPolygon = shapely.affinity.scale(
      polygons, xfact=normal_coordinates_to_dataset_coordinates(1, 0, height, width, x_max, y_min)[0],
      yfact=normal_coordinates_to_dataset_coordinates(0, 1, height, width, x_max, y_min)[1],
      origin=(0, 0, 0))

    # for rounding, which is necessary to avoid eval errors
    row[1]['MultipolygonWKT'] = __get_valid_wkt_str(MultiPolygon(multiPolygon), 6)
    #row[1]['MultipolygonWKT'] = __get_valid_wkt_str(loads(row[1]['MultipolygonWKT']))
    f.write(",".join(
      [str(row[1]['ImageId']), str(row[1]['ClassType']), '"' + str(row[1]['MultipolygonWKT']) + '"']) + '\n')



def mask_to_polygons(mask, epsilon=5, min_area=1.):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly

    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons



if __name__ == "__main__":

  process_submission()
