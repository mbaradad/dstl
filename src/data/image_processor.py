from utils.dirs import *
import pandas as pd
from shapely.wkt import loads
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from utils.utils import resize
from matplotlib.collections import PatchCollection
from tifffile import tifffile

'''
Classes of masks
1. Buildings
2. Misc. Manmade structures
3. Road
4. Track - poor/dirt/cart track, footpath/trail
5. Trees - woodland, hedgerows, groups of trees, standalone trees
6. Crops - contour ploughing/cropland, grain (wheat) crops, row (potatoes, turnips) crops
7. Waterway
8. Standing water
9. Vehicle Large - large vehicle (e.g. lorry, truck,bus), logistics vehicle
10. Vehicle Small - small vehicle (car, van), motorbike
'''

class ImageProcessor():
  def __init__(self):
    self.df_wkt = pd.read_csv(TRAIN_WKT)
    self.grid_sizes = pd.read_csv(GRID_SIZES)
    self.class_types = 10
  #TODO: possibly normalize images at this point or at other point
  #TODO: check whole precisison is preserved
  #TODO: everything is resized to image size, which is the maximum size, though some channels (i.e infrared) are of lower size.
  def get_images(self, name):
    images = tifffile.imread(THREE_BAND + '/' + name + '.tif')
    for i in 'A','M','P':
      image = tifffile.imread(SIXTEEN_BAND + '/' + name + '_' + i + '.tif')
      #resize is specified in the opposite order than np.shape!
      if len(image.shape) == 2:
        image = np.expand_dims(image, 0)
      image = np.transpose(image, [1,2,0])
      image = resize(image, images[0].shape[0], images[0].shape[1])
      image = np.transpose(image, [2, 0, 1])
      images = np.append(images, image, axis=0)
    return images

  # TODO: maybe do it more efficiently, without requiring pyplot
  # though this is only done once
  # also maybe store the masks directly to disk
  def get_masks(self, idx, height, width):

    polygonsList = {}
    image = self.df_wkt[self.df_wkt.ImageId == idx]
    masks = list()
    #for cType in range(self.class_types):
    for cType in range(4,5):
      #TODO: CHECK if image has been previously preprocessed, and use that, once we know for sure that there are no errors
      polygonsList = loads(image[image.ClassType == (cType + 1)].MultipolygonWKT.values[0])
      if len(polygonsList) == 0:
        masks.append(np.zeros([1, height, width]))
        continue

      #plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
      # fig = plt.figure()
      #plt.axis('off')
      #plt.imshow(np.zeros([height, width, 3], dtype=np.uint8))

      #ax = plt.gca()
      #ax.set_autoscale_on(False)



      polygonsList = {}
      for cType in image.ClassType.unique():
        polygonsList[cType] = loads(image[image.ClassType == cType].MultipolygonWKT.values[0])

      # plot using matplotlib

      fig, ax = plt.subplots(figsize=(8, 8))

      # plotting, color by class type
      for p in polygonsList:
        for polygon in polygonsList[p]:
          mpl_poly = Polygon(np.array(polygon.exterior), color=plt.cm.Set1(p * 10), lw=0, alpha=0.3)
          ax.add_patch(mpl_poly)

      ax.relim()
      ax.autoscale_view()

      c = (np.ones((1, 3))).tolist()[0]
      polygons = list()
      color = list()
      fig, ax = plt.subplots(figsize=(8, 8))

      for polygon in polygonsList:
        mpl_poly = Polygon(np.array(polygon.exterior), color=plt.cm.Set1(cType * 10), lw=0, alpha=0.3)
        ax.add_patch(mpl_poly)
        polygons.append(Polygon(np.array(polygon.exterior)))
        color.append(c)

      p = PatchCollection(polygons, facecolor=color)
      ax.add_collection(p)
      extent = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())


      plt.savefig(PREPROCESSED_INPUT + '/tmp.png', bbox_inches=extent)

      plt.close()
      mask2 = cv2.imread(PREPROCESSED_INPUT + '/tmp.png')
      mask2 = resize(mask2 , height, width)

      masks.append(mask2[:, :, 0].astype("bool"))
    return np.asarray(masks)