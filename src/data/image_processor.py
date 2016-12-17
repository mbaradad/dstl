from utils.dirs import *
import pandas as pd
from shapely.wkt import loads
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

class ImageProcessor():
  def __init__(self):
    self.df_wkt = pd.read_csv(TRAIN_WKT)
    self.grid_sizes = pd.read_csv(GRID_SIZES)

  #TODO: possibly normalize images at this point or at other point
  #TODO: check whole precisison is preserved
  #TODO: everything is resized to image size, which is the maximum size, though some channels (i.e infrared) are of lower size.
  def get_images(self, name):
    image = cv2.imread(THREE_BAND + '/' + name + '.tif')
    images = np.transpose(image, [2, 0, 1])
    for i in 'A','M','P':
      image = cv2.imread(SIXTEEN_BAND + '/' + name + '_' + i + '.tif')
      #resize is specified in the opposite order than np.shape!
      image = cv2.resize(image, (images[0].shape[1],images[0].shape[0]))
      image = np.transpose(image, [2, 0, 1])
      np.append(images, image, axis=0).shape
    return images

  # TODO: maybe do it more efficiently, without requiring pyplot
  # though this is only done once
  # also maybe store the masks directly to disk
  def get_masks(self, idx):
      polygonsList = {}
      image = self.df_wkt[self.df_wkt.ImageId == idx]
      for cType in image.ClassType.unique():
        polygonsList[cType] = loads(image[image.ClassType == cType].MultipolygonWKT.values[0])
      fig, ax = plt.subplots(figsize=(9, 9))
      for p in polygonsList:
        for polygon in polygonsList[p]:
          mpl_poly = Polygon(np.array(polygon.exterior), color=plt.cm.Set1(p * 10), lw=0, alpha=0.3)
          ax.add_patch(mpl_poly)
      ax.relim()
      ax.autoscale_view()
      plt.title(idx)