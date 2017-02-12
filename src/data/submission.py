from dataset import Dataset
from classifiers.resnet_keras.classifier import ResnetClassifier, MultipleClassifier
import pycocotools.mask as mask
import numpy as np
from utils.utils import *
from shapely.affinity import scale
from sima.ROI import mask2poly, poly2mask
from utils.utils import normal_coordinates_to_dataset_coordinates
from shapely.ops import cascaded_union
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.collection import GeometryCollection
from shapely.geometry.polygon import Polygon
import shapely
import datetime
from utils.dirs import *

import tensorflow as tf
import cv2
from collections import defaultdict
import pandas as pd

import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import math
import png
from repair_topology import __get_valid_wkt_str

def calc_precision(x, y):
  if not x: return 0
  power_x = -int(math.floor(math.log10(abs(x))))
  power_y = -int(math.floor(math.log10(abs(y))))
  return max(power_x, power_y)

def add_prediction(masks, pos, crop_size, predictions, overlapping_percentage):

  predicted_mask = np.zeros(masks.shape)
  predicted_mask[:, pos[1]:(pos[1] + crop_size[0]), pos[2]:(pos[2] + crop_size[1])] = predictions

  kernel = np.ones((int(crop_size[0] * overlapping_percentage), int(crop_size[1] * overlapping_percentage)), np.float32)

  eroded_mask = cv2.erode(np.asarray(predicted_mask[0] != 0, np.float32), kernel)
  indexes = np.where(eroded_mask != 0)

  masks[:,indexes[0][0]:indexes[0][-1], indexes[1][0]:indexes[1][-1]] = predicted_mask[:,indexes[0][0]:indexes[0][-1], indexes[1][0]:indexes[1][-1]]
  return masks


def generate_submission(classifier):
  #kernel_closings = [5, 5, 5, 5, 10, 5, 5, 5, 1, 1]
  #kernel_openings = [5, 5, 5, 5, 10, 5, 5, 5, 1, 1]

  #Better to process afterwards
  kernel_closings = [5, 5, 10, 20, 10, 10, 10, 10, 10, 10]
  kernel_openings = [10, 6, 7, 7, 10, 10, 2, 2, 2, 2]
  #kernel_closings = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  #kernel_openings = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


  d = Dataset(train=False)
  now = datetime.datetime.now()
  f = open(SUBMISSION + '/submission_' + str(now).replace(" ", "_") + '.csv', 'w')
  f_original = open(SUBMISSION + '/submission_' + str(now).replace(" ", "_") + '_original.csv', 'w')

  images_dir = '/mnt/sdd1/submissions/submission_' + str(now).replace(" ", "_")
  mkdir(images_dir)
  f.write(get_header() + '\n')  # python will convert \n to os.linesep

  chunk_size = 16
  crop_size = classifier.get_crop_size()
  overlapping_percentage = 0.15
  idxs = d.get_generator_idxs(crop_size=crop_size, subset="", overlapping_percentage=overlapping_percentage)
  positions_by_idx = dict()
  for idx in idxs:
    if idx[0] in positions_by_idx.keys():
      positions_by_idx[idx[0]].append(idx)
    else:
      positions_by_idx[idx[0]] = [idx]
  #df = pd.read_csv('/home/manel/Documents/dstl/output/submissions/submission_2017-01-31_23:53:38.265697.csv')
  for idx in range(len(d.get_image_list())):
    print 'Predicting image ' + str(idx + 1) + '/' + str(len(d.get_image_list()))
    '''
    #For solving problems with some instances
    if d.get_image_list()[idx] != '6110_2_3':
      for i in range(10):
        f.write(",".join([str(df.ix[idx*10+i][0]), str(df.ix[idx*10+i][1]), '"' + str(df.ix[idx*10+i][2]) + '"']) + '\n')
      continue

    for i in range(10):
      if i == 4:
        f.write(",".join(
          [str(df.ix[idx * 10 + i][0]), str(df.ix[idx * 10 + i][1]), 'MULTIPOLYGON EMPTY']) + '\n')
      else:
        f.write(",".join([str(df.ix[idx*10+i][0]), str(df.ix[idx*10+i][1]), '"' + str(df.ix[idx*10+i][2]) + '"']) + '\n')
    continue
    '''

    masks = np.zeros([10, d.image_sizes[idx][0], d.image_sizes[idx][1]], dtype='float32')
    d.generate_one(idx)
    for i in range(0, len(positions_by_idx[idx]), chunk_size):

      actual_pos = positions_by_idx[idx][i:i+chunk_size]
      ims = np.zeros([chunk_size, 20, crop_size[0], crop_size[1]])
      j = 0
      for pos in actual_pos:
        im = d.generate_one_cropped(pos[0], crop_size, pos[1], pos[2])[0]
        ims[j] = im
        j+=1
      predictions = classifier.predict([ims[:,:3,:,:], ims[:,3:,:,:]])
      j = 0
      for pos in actual_pos:
        masks = add_prediction(masks, pos, crop_size, predictions[j], overlapping_percentage)
        j+=1
    height = masks.shape[1]
    width = masks.shape[2]
    [x_max, y_min] = d.get_grid_size(idx)
    for i in range(len(masks)):
      print 'Computing mask' + str(i + 1)
      '''
      if i in masks_from_previous_df:
        print 'Mask ' + str(i + 1) + 'from df'
        f.write(",".join([str(df.ix[idx*10+i][0]), str(df.ix[idx*10+i][1]), '"' + str(df.ix[idx*10+i][2]) + '"']) + '\n')
      if idx < 4 and generate_images:
        plt.imshow(masks[i])
        plt.savefig(images_dir +'/img_' + str(idx) + '_m_' + str(i) + '_' + d.class_id_to_name(i) +'.jpg')

      My method, changed f
      actual_mask = cv2.morphologyEx(np.asarray(masks[i], dtype='float32'), cv2.MORPH_OPEN, kernel)
      actual_mask = cv2.morphologyEx(actual_mask, cv2.MORPH_OPEN, kernel)
      actual_mask = np.asarray(actual_mask, dtype='bool')
      polygons = mask2poly(actual_mask)
      polygons_union = cascaded_union(polygons)
      if type(polygons_union) is MultiPolygon or type(polygons_union) is GeometryCollection:
        polygons = polygons_union
      else:
        polygons = list()
        polygons.append(polygons_union)
            for polygon in polygons:
        transformed_coordinates = [normal_coordinates_to_dataset_coordinates(x, y)
                                   for (x, y) in np.array(polygon.exterior)]
        normalized_polygons.append(Polygon(transformed_coordinates))
      multiPolygon = MultiPolygon(normalized_polygons)
      '''

      #polygons = mask_to_polygons(masks[i], epsilon=30, min_area=10)

      np.savez_compressed(images_dir +'/mask_' + d.get_image_list()[idx] + '_' + str(i) + '.h5', masks[i])
      actual_mask = np.asarray(masks[i] > 0.5, dtype='uint8')

      polygons = mask_to_polygons(actual_mask, epsilon=5, min_area=5)
      f_original.write(d.get_image_list()[idx] + ',' + str(i + 1) + ',"' + __get_valid_wkt_str(polygons, 6) + '"\n')

      if kernel_closings[i] != 0:
        kernel_close = np.ones((kernel_closings[i], kernel_closings[i]), np.uint8)
        actual_mask = cv2.morphologyEx(actual_mask, cv2.MORPH_CLOSE, kernel_close)
        kernel_open = np.ones((kernel_openings[i], kernel_openings[i]), np.uint8)
        actual_mask = cv2.morphologyEx(actual_mask, cv2.MORPH_OPEN, kernel_open)
      #print 'Computing cv2 open/close'
      polygons = mask_to_polygons(actual_mask, epsilon=5, min_area=5)

      '''
      normalized_polygons = list()
      for polygon in polygons:
        transformed_coordinates = [normal_coordinates_to_dataset_coordinates(x, y, height, width, x_max, y_min)
                                   for (x, y) in np.array(polygon.exterior)]
        normalized_polygons.append(Polygon(transformed_coordinates))
      multiPolygon = MultiPolygon(normalized_polygons)
      '''

      multiPolygon = shapely.affinity.scale(
        polygons, xfact=normal_coordinates_to_dataset_coordinates(1, 0, height, width, x_max, y_min)[0],
        yfact=normal_coordinates_to_dataset_coordinates(0, 1, height, width, x_max, y_min)[1],
        origin=(0, 0, 0))


      #for rounding, which is necessary to avoid eval errors
      precision = calc_precision(normal_coordinates_to_dataset_coordinates(1, 0, height, width, x_max, y_min)[0],
                                 normal_coordinates_to_dataset_coordinates(0, 1, height, width, x_max, y_min)[1])

      f.write(d.get_image_list()[idx] + ',' + str(i + 1) + ',"' + __get_valid_wkt_str(multiPolygon, 6) + '"\n')
      #f.write(d.get_image_list()[idx] + ',' + str(i + 1) + ',"' + shapely.wkt.dumps(multiPolygon) + '"\n')
    #for i in range(len(masks)):
    #open and close masks to simplify image:    kernel = np.ones((10, 10), np.uint8)


  f.close()

def mask_to_polygons_rasterio(mask):
  with rasterio.drivers():
    results = (
      {'properties': {'raster_val': v}, 'geometry': s}
      for i, (s, v)
      in enumerate(
      shapes(mask)))
  return cascaded_union([shape(d['geometry']) for d in list(results) if d['properties']['raster_val'] == 1])

def get_scalers(h, w, x_max, y_min):
  w = float(w)
  h = float(h)
  w_ = w * (w / (w + 1))
  h_ = h * (h / (h + 1))
  return w_ / x_max, h_ / y_min


def mask_to_polygons(mask, epsilon=10., min_area=10.):
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
  all_polygons = all_polygons.buffer(-1)
  all_polygons = MultiPolygon(all_polygons)
  return all_polygons



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
        all_polygons = all_polygons.buffer(-1)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons




def get_header():
  return "ImageId,ClassType,MultipolygonWKT"

if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
  classifier = ResnetClassifier(RESNET_KERAS_OUTPUT + "/execution_2017-02-1202:49:27.328895/model.h5")
  #classifier = MultipleClassifier()
  generate_submission(classifier)

